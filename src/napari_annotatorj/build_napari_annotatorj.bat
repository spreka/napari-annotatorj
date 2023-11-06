echo "removing old files..."
rmdir /s /q build
rmdir /s /q pyinstaller


echo "building app napari_annotatorj..."

SET scriptpath=%~dp0
pyinstaller --noconfirm --clean --distpath pyinstaller --log-level=INFO "%scriptpath%\napari_annotatorj.spec"
