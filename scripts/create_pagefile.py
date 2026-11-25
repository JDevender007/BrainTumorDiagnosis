#!/usr/bin/env python3
"""Restore a 29 GB pagefile so the commit ceiling goes 15.3 GB -> ~44 GB (run ELEVATED).

What it does:
1. Writes the persistent registry entry HKLM\\...\\Memory Management\\PagingFiles
   = "D:\\pagefile.sys 29696 32768" (needs Administrators FullControl - present).
2. Attempts a runtime attach via NtCreatePagefile (best effort; NOT exported on
   this Windows build, so this is expected to report "absent").
3. Reads the commit ceiling before/after and decides honestly:
   - ceiling >= 40 GB           -> success=True, reboot_required=False
   - registry written, ceiling unchanged -> success=False, action_required="reboot"

Writes results/phase3/pagefile_create.log.json; safe to re-run; deletes nothing.
"""
import ctypes
import json
import time
from ctypes import wintypes

LOG_PATH = r"D:\BrainTumorDiagnosis\results\phase3\pagefile_create.log.json"
PAGEFILE_PATH = "D:\\pagefile.sys"
INIT_MB = 29696   # 29 GB  -> ceiling 15.3 + 29 = 44.3 GB
MAX_MB = 32768    # 32 GB  -> headroom so the commit-then-free growth trick still works


def memstatus():
    class MI(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    m = MI()
    m.dwLength = ctypes.sizeof(MI)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {
        "ceiling_gb": round(m.ullTotalPageFile / 2**30, 2),
        "avail_gb": round(m.ullAvailPageFile / 2**30, 2),
        "ram_total_gb": round(m.ullTotalPhys / 2**30, 2),
    }


class UNICODE_STRING(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.USHORT),
        ("MaximumLength", wintypes.USHORT),
        ("Buffer", wintypes.LPWSTR),
    ]


class OBJECT_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.ULONG),
        ("RootDirectory", wintypes.HANDLE),
        ("ObjectName", ctypes.POINTER(UNICODE_STRING)),
        ("Attributes", wintypes.ULONG),
        ("SecurityDescriptor", ctypes.c_void_p),
        ("SecurityQualityOfService", ctypes.c_void_p),
    ]


def nt_create_pagefile(path, init_mb, max_mb, desired_access):
    """NtCreatePagefile -> (status_int, handle_or_None). status >= 0 is success."""
    ntdll = ctypes.windll.ntdll
    fn = ntdll.NtCreatePagefile  # AttributeError if not exported on this build
    fn.restype = ctypes.c_long
    fn.argtypes = [
        ctypes.POINTER(wintypes.HANDLE), wintypes.ULONG, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, wintypes.ULONG,
    ]
    buf = ctypes.create_unicode_buffer(path)
    us = UNICODE_STRING()
    us.Buffer = buf
    us.Length = len(path) * 2
    us.MaximumLength = (len(path) + 1) * 2
    oa = OBJECT_ATTRIBUTES()
    oa.Length = ctypes.sizeof(OBJECT_ATTRIBUTES)
    oa.ObjectName = ctypes.pointer(us)
    oa.Attributes = 0x40  # OBJ_CASE_INSENSITIVE
    handle = wintypes.HANDLE()
    maxsz = ctypes.c_int64(max_mb * 1024 * 1024)
    initsz = ctypes.c_int64(init_mb * 1024 * 1024)
    status = fn(
        ctypes.byref(handle), desired_access, ctypes.byref(oa),
        ctypes.byref(maxsz), ctypes.byref(initsz), 0,
    )
    return status, (handle if status >= 0 else None)


def write_registry(log):
    """Write PagingFiles. Needs read+write on the key (Administrators FullControl)."""
    import winreg
    key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, key_path, 0,
            winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE,
        ) as k:
            old = winreg.QueryValueEx(k, "PagingFiles")[0]
            winreg.SetValueEx(
                k, "PagingFiles", 0, winreg.REG_MULTI_SZ,
                [f"{PAGEFILE_PATH} {INIT_MB} {MAX_MB}"],
            )
            verify = winreg.QueryValueEx(k, "PagingFiles")[0]
        log["actions"].append({"registry": "PagingFiles set", "old_value": old,
                               "verify": verify})
        return True
    except OSError as e:
        log["actions"].append({"registry": f"FAILED {type(e).__name__}: {e}"})
        return False


def runtime_attach(log):
    """Best effort: create the pagefile NOW without a reboot."""
    attempts = []
    path_forms = (r"\??\D:\pagefile.sys", "D:\\pagefile.sys")
    try:
        for pform in path_forms:
            for access, label in ((0x1F01FF, "FILE_ALL_ACCESS"), (0x10000000, "GENERIC_ALL")):
                try:
                    status, h = nt_create_pagefile(pform, INIT_MB, MAX_MB, access)
                    attempts.append({
                        "path": pform, "access": label,
                        "status": f"0x{status & 0xFFFFFFFF:08X}", "ok": status >= 0,
                    })
                    if status >= 0:
                        if h:
                            ctypes.windll.kernel32.CloseHandle(h)
                        break
                except OSError as e:
                    attempts.append({"path": pform, "access": label, "error": str(e)})
            else:
                continue
            break
    except AttributeError as e:
        attempts.append({"error": f"NtCreatePagefile unavailable: {e}",
                         "note": "expected on this build - reboot path is used"})
    log["actions"].append({"nt_create_pagefile": attempts})
    return any(a.get("ok") for a in attempts if isinstance(a, dict))


def main():
    log = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "actions": []}
    try:
        log["elevated"] = bool(ctypes.windll.shell32.IsUserAnAdmin())
        log["before"] = memstatus()

        registry_ok = write_registry(log)
        attached = runtime_attach(log)

        time.sleep(1)
        log["after"] = memstatus()
        try:
            import os
            log["pagefile_file"] = {
                "exists": os.path.exists(PAGEFILE_PATH),
                "size_gb": round(os.path.getsize(PAGEFILE_PATH) / 2**30, 2)
                if os.path.exists(PAGEFILE_PATH) else 0,
            }
        except OSError as e:
            log["pagefile_file"] = {"error": str(e)}

        ceiling_now = log["after"]["ceiling_gb"]
        if ceiling_now >= 40:
            log["success"] = True
            log["action_required"] = "none"
        elif registry_ok:
            log["success"] = False
            log["action_required"] = "reboot"
            log["note"] = ("PagingFiles written; pagefile is created by the session "
                           "manager at boot -> reboot to reach ~44 GB commit")
        else:
            log["success"] = False
            log["action_required"] = "manual_registry_fix"
        log["reboot_required"] = ceiling_now < 20 and registry_ok
    except Exception as e:
        log["error"] = f"{type(e).__name__}: {e}"
        log["success"] = False
        log.setdefault("action_required", "inspect_log")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
