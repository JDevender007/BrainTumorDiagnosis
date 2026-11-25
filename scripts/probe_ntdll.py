#!/usr/bin/env python3
"""Definitive diagnostics: ntdll export scan for pagefile APIs + registry ACL.

Writes results/phase3/pagefile_probe.json. Read-only; changes nothing.
"""
import ctypes
import json
import struct

OUT = r"D:\BrainTumorDiagnosis\results\phase3\pagefile_probe.json"


def read_pe_exports(path):
    with open(path, "rb") as f:
        data = f.read()
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    # PE32+ only (ntdll on x64)
    magic = struct.unpack_from("<H", data, e_lfanew + 24)[0]
    assert magic == 0x20B, f"unexpected optional magic {magic:#x}"
    export_rva = struct.unpack_from("<I", data, e_lfanew + 24 + 112)[0]
    if export_rva == 0:
        return []
    # section table for RVA->offset
    num_sections = struct.unpack_from("<H", data, e_lfanew + 6)[0]
    opt_size = struct.unpack_from("<H", data, e_lfanew + 20)[0]
    sec_off = e_lfanew + 24 + opt_size
    sections = []
    for i in range(num_sections):
        off = sec_off + i * 40
        va = struct.unpack_from("<I", data, off + 12)[0]
        vsz = struct.unpack_from("<I", data, off + 8)[0]
        raw = struct.unpack_from("<I", data, off + 20)[0]
        rsz = struct.unpack_from("<I", data, off + 16)[0]
        sections.append((va, vsz, raw, rsz))

    def rva2off(rva):
        for va, vsz, raw, rsz in sections:
            if va <= rva < va + max(vsz, rsz):
                return raw + (rva - va)
        return None

    eo = rva2off(export_rva)
    num_names = struct.unpack_from("<I", data, eo + 24)[0]
    names_rva = struct.unpack_from("<I", data, eo + 32)[0]
    no = rva2off(names_rva)
    names = []
    for i in range(num_names):
        nrva = struct.unpack_from("<I", data, no + i * 4)[0]
        noff = rva2off(nrva)
        end = data.index(b"\x00", noff)
        names.append(data[noff:end].decode("ascii", "replace"))
    return names


def main():
    result = {}
    try:
        names = read_pe_exports(r"C:\Windows\System32\ntdll.dll")
        result["ntdll_export_count"] = len(names)
        result["pagefile_exports"] = sorted(n for n in names if "agefile" in n.lower())
        result["has_NtCreatePagefile"] = "NtCreatePagefile" in names
        # sanity: known-good export must be present
        result["sanity_NtAllocateVirtualMemory"] = "NtAllocateVirtualMemory" in names
        result["systeminfo_exports"] = sorted(
            n for n in names if n.startswith("NtSetSystemInformation")
        )
    except Exception as e:
        result["export_scan_error"] = f"{type(e).__name__}: {e}"

    # runtime ctypes lookup (what the script actually does)
    try:
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        lookup = {}
        for nm in ("NtCreatePagefile", "NtAllocateVirtualAddress" , "NtAllocateVirtualMemory",
                   "NtSetSystemInformation"):
            try:
                getattr(ntdll, nm)
                lookup[nm] = "present"
            except AttributeError:
                lookup[nm] = "absent"
        result["ctypes_lookup"] = lookup
    except Exception as e:
        result["ctypes_error"] = f"{type(e).__name__}: {e}"

    # registry ACL + open rights test
    import winreg
    key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
    rights = {}
    for acc, label in ((winreg.KEY_QUERY_VALUE, "QUERY"),
                       (winreg.KEY_SET_VALUE, "SET"),
                       (winreg.KEY_READ, "READ"),
                       (winreg.KEY_WRITE, "WRITE"),
                       (winreg.KEY_ALL_ACCESS, "ALL")):
        try:
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, acc)
            winreg.CloseKey(k)
            rights[label] = "OK"
        except OSError as e:
            rights[label] = f"FAIL {e.winerror} {e}"
    result["registry_open_rights"] = rights
    try:
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0,
                           winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE)
        old = winreg.QueryValueEx(k, "PagingFiles")[0]
        winreg.CloseKey(k)
        result["registry_read_with_QUERY_SET"] = {"ok": True, "old_value": old}
    except OSError as e:
        result["registry_read_with_QUERY_SET"] = {"ok": False, "error": str(e)}

    # is this process elevated?
    try:
        result["is_admin"] = bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        pass

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
