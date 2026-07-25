#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Activating venv ==="
source venv/bin/activate

echo "=== Cleaning old build ==="
rm -rf build dist

# py2app сам ад-хок подписывает payload-dylib (portaudio/sndfile), а затем macholib
# переписывает в них install-names. На arm64 ядро (AMFI) периодически блокирует запись
# в уже подписанный Mach-O (EPERM), и py2app падает недетерминированно. Лечим ретраем:
# чистая пересборка до 5 раз, пока не пройдёт (источники в venv не трогаем).
echo "=== Building MacDictator.app (retry on transient codesign EPERM) ==="
built=0
for attempt in 1 2 3 4 5; do
    rm -rf build dist
    if python setup.py py2app > /tmp/macdictator_py2app.log 2>&1; then
        built=1; break
    fi
    if grep -q 'Operation not permitted' /tmp/macdictator_py2app.log; then
        echo "  attempt $attempt: transient EPERM on a signed dylib, retrying..."
        sleep 2
    else
        echo "  py2app failed (not the EPERM race):"; tail -30 /tmp/macdictator_py2app.log
        exit 1
    fi
done
[ "$built" = 1 ] || { echo "ERROR: py2app still hitting EPERM after 5 attempts" >&2; exit 1; }

APP=dist/MacDictator.app

# py2app тащит только dylib'ы Tcl/Tk, но не скрипты (init.tcl, tk.tcl и пр.).
# Без них .app падает на старте на любой машине без Homebrew tcl-tk.
# Кладём каталоги скриптов в Resources/lib — app.py укажет на них через TCL_LIBRARY.
echo "=== Bundling Tcl/Tk script libraries ==="
TCLTK_LIB=/opt/homebrew/opt/tcl-tk/lib
mkdir -p "$APP/Contents/Resources/lib"
cp -R "$TCLTK_LIB/tcl9.0" "$APP/Contents/Resources/lib/tcl9.0"
cp -R "$TCLTK_LIB/tk9.0"  "$APP/Contents/Resources/lib/tk9.0"
# Homebrew раздаёт файлы read-only — снимаем, чтобы codesign/очистка не спотыкались.
chmod -R u+w "$APP/Contents/Resources/lib/tcl9.0" "$APP/Contents/Resources/lib/tk9.0"

# Подпись стабильной идентити (issue #5): ad-hoc подпись меняет CDHash каждой
# сборкой, и TCC сбрасывает Accessibility-грант при любом обновлении. С
# сертификатом «MacDictator Dev» designated requirement стабилен — грант
# выдаётся один раз. Нет сертификата в keychain — откат на ad-hoc как раньше.
SIGN_ID="${MACDICTATOR_SIGN_ID:-MacDictator Dev}"
if security find-identity -p codesigning 2>/dev/null | grep -q "$SIGN_ID"; then
    SIGN="$SIGN_ID"
    echo "=== Signing with identity: $SIGN_ID ==="
else
    SIGN="-"
    echo "=== Signing ad-hoc ($SIGN_ID not found in keychain; грант Accessibility слетит) ==="
fi

# py2app signs binaries before macholib rewrites install names, which corrupts
# signatures; on arm64 the kernel then SIGKILLs the app (Code Signature Invalid).
# Re-sign every Mach-O bottom-up, then the bundle itself.
echo "=== Re-signing bundle ==="
find "$APP" -type f \( -name "*.so" -o -name "*.dylib" \) -exec codesign --force --sign "$SIGN" {} \; 2>/dev/null

# Some Homebrew dylibs (liblzma) get mangled beyond what codesign can replace.
# Re-copy the original and rewrite its install id, then sign.
for f in "$APP/Contents/Frameworks/"*.dylib; do
    codesign --verify "$f" 2>/dev/null && continue
    name=$(basename "$f")
    src=$(find -L /opt/homebrew/opt -name "$name" -type f 2>/dev/null | head -1)
    if [ -z "$src" ]; then
        echo "ERROR: $name is corrupt and no Homebrew original found" >&2
        exit 1
    fi
    echo "replacing corrupt $name from $src"
    cp -f "$src" "$f"
    chmod u+w "$f"
    install_name_tool -id "@executable_path/../Frameworks/$name" "$f" 2>/dev/null
    codesign --force --sign "$SIGN" "$f"
    codesign --verify "$f" || { echo "ERROR: $name still invalid" >&2; exit 1; }
done
for bin in "$APP/Contents/Resources/lib/python3.12/torch/bin/"*; do
    [ -f "$bin" ] && codesign --force --sign "$SIGN" "$bin" 2>/dev/null
done
codesign --force --sign "$SIGN" "$APP/Contents/Frameworks/Python.framework/Versions/3.12/Python" 2>/dev/null
codesign --force --sign "$SIGN" "$APP/Contents/Frameworks/Python.framework/Versions/3.12" 2>/dev/null
codesign --force --sign "$SIGN" "$APP/Contents/MacOS/python"
codesign --force --sign "$SIGN" "$APP/Contents/MacOS/MacDictator"
codesign --force --sign "$SIGN" "$APP"

echo "=== Verifying signature ==="
codesign --verify --deep --strict "$APP"

echo "=== Building DMG ==="
VERSION=$(grep CFBundleShortVersionString setup.py | head -1 | sed -E "s/.*'([0-9.]+)'.*/\1/")
DMG="MacDictator-${VERSION}.dmg"
rm -f "$DMG"
STAGING=$(mktemp -d)
ditto "$APP" "$STAGING/MacDictator.app"   # ditto сохраняет подписи и xattr
ln -s /Applications "$STAGING/Applications"
hdiutil create -volname "MacDictator" -srcfolder "$STAGING" -ov -format UDZO "$DMG"
rm -rf "$STAGING"

echo ""
echo "=== Done! ==="
echo "App: dist/MacDictator.app"
echo "DMG: $DMG"
echo ""
echo "To run:  open dist/MacDictator.app"
