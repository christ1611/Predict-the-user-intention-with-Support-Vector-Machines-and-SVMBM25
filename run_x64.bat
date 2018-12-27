@echo off
setlocal
set MINGW64_HOME=D:\work\binary\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64
set PATH=%PATH%;%MINGW64_HOME%\bin
del SVMBM25.exe
copy /V /Y Debug\SVMBM25.exe .
SVMBM25.exe
endlocal