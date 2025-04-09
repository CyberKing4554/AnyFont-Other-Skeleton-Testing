Dim objShell
Set objShell = WScript.CreateObject("WScript.Shell")
command = "powershell.exe -Command ""& python ./gui.py"""
objShell.Run command, 1, True
Set objShell = Nothing