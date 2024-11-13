[app]

# (str) Title of your application
title = My Application

# (str) Package name
package.name = myapp

# (str) Package domain (needed for android/ios packaging)
package.domain = org.test

# (str) Source code where the main.py lives
source.dir = .

# (list) Source files to include (leave empty to include all files)
source.include_exts = py,png,jpg,kv,atlas

# (list) List of inclusions using pattern matching
# source.include_patterns = assets/*,images/*.png

# (list) Source files to exclude (leave empty to not exclude anything)
# source.exclude_exts = spec

# (list) List of directory to exclude (leave empty to not exclude anything)
# source.exclude_dirs = tests, bin, venv

# (list) List of exclusions using pattern matching
# source.exclude_patterns = license,images/*/*.jpg

# (str) Application versioning (method 1)
version = 0.1

# (list) Application requirements
# Include necessary libraries for your app
requirements = python3==3.7.6,hostpython3==3.7.6,kivy,pillow

# (str) Presplash of the application (optional)
# presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application (optional)
# icon.filename = %(source.dir)s/data/icon.png

# (list) Supported orientations
# Valid options are: landscape, portrait, portrait-reverse, or landscape-reverse
orientation = portrait

# (list) List of service to declare (optional)
# services = NAME:ENTRYPOINT_TO_PY,NAME2:ENTRYPOINT2_TO_PY

# OSX Specific

# (str) Version of Python to use on OSX
osx.python_version = 3.7.6

# Kivy version to use on OSX
osx.kivy_version = 1.9.1

# Android specific

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 1

# (string) Presplash background color (for Android toolchain)
# android.presplash_color = #FFFFFF

# (str) Adaptive icon for Android (API 26+)
# icon.adaptive_foreground.filename = %(source.dir)s/data/icon_fg.png
# icon.adaptive_background.filename = %(source.dir)s/data/icon_bg.png

# (list) Permissions for Android (optional)
# android.permissions = android.permission.INTERNET, android.permission.WRITE_EXTERNAL_STORAGE

# (int) Target Android API level, should be as high as possible
android.api = 31

# (int) Minimum API your APK will support
android.minapi = 21

# (str) Android NDK version to use
android.ndk = 23b

# (int) Android NDK API to use; usually matches android.minapi
android.ndk_api = 21

# (bool) Accept SDK licenses automatically (recommended for CI builds)
android.accept_sdk_license = True

# (str) Android entry point, default is ok for Kivy-based app
# android.entrypoint = org.kivy.android.PythonActivity

# (list) Pattern to whitelist for the whole project
# android.whitelist =

# (str) Path to a custom whitelist file
# android.whitelist_src =

# (list) Java classes to add as activities to the manifest
# android.add_activities = com.example.ExampleActivity

# (list) Gradle dependencies to add
# android.gradle_dependencies =

# (bool) Enable AndroidX support (requires android.api >= 28)
# android.enable_androidx = True

# (list) Java compile options for gradle
# android.add_compile_options = "sourceCompatibility = 1.8", "targetCompatibility = 1.8"

# (list) Gradle repositories to add
# android.add_gradle_repositories = "maven { url 'https://kotlin.bintray.com/ktor' }"

# (list) Packaging options to add for gradle
# android.add_packaging_options = "exclude 'META-INF/common.kotlin_module'", "exclude 'META-INF/*.kotlin_module'"

# (str) OUYA Console category (GAME or APP)
# android.ouya.category = GAME

# (list) Copy these files or directories in the apk assets directory
# android.add_assets =

# (list) Resources to add (images, xml, etc.)
# android.add_resources = legal_icons:drawable

# (bool) Indicate whether the screen should stay on
# android.wakelock = False

# (list) Android application meta-data
# android.meta_data =

# (str) Android logcat filters to use
# android.logcat_filters = *:S python:D

# (bool) Copy library instead of making a libpymodules.so
# android.copy_libs = 1

# (list) Android architectures to build for
android.archs = arm64-v8a, armeabi-v7a

# (bool) Enables Android auto-backup feature (for Android API >= 23)
android.allow_backup = True

# (str) Format to package the app for debug mode
android.debug_artifact = apk

# (str) Format to package the app for release mode (optional)
# android.release_artifact = aab

# Python for android (p4a) specific

# (str) python-for-android URL to use for checkout
# p4a.url =

# (str) python-for-android fork to use in case if p4a.url is not specified
# p4a.fork = kivy

# (str) python-for-android branch to use
# p4a.branch = master

# (str) Bootstrap to use for android builds
# p4a.bootstrap = sdl2

[ios]

# (str) Path to a custom kivy-ios folder
ios.kivy_ios_url = https://github.com/kivy/kivy-ios
ios.kivy_ios_branch = master

# (str) Name of the certificate to use for signing the debug version
# ios.codesign.debug = "iPhone Developer: <lastname> <firstname> (<hexstring>)"

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug)
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1

