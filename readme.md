# MacOS Configuration
The configuration is for
1. Install boost libs: `brew install boost-python3` (My current version is 1.73.0)
2. Install JUCE (the version should be JUCE 5)
3. Make sure have python3.8 installed
4. Get RenderMan from https://github.com/fedden/RenderMan
5. Open `RenderMan-py36.jucer` in the project folder, generate XCode project.
6. Configure the search paths in XCode (This could be different, find the corresponding paths on your machine):
   - Boost dylib path: `/usr/local/Cellar/boost-python3/1.73.0/lib`
   - Python dylib path: `/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/lib`
   - Python header path: `/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/include/python3.8`
7. Set the link flags: `-shared -lboost_python38 -undefined dynamic_lookup`
7. Build a release build `librenderman.so.dylib`, remove the extra `.dylib` extension.
8. Copy the `librenderman.so` to your python project root directory.

# Example
[RenderMan Example](http://doc.gold.ac.uk/~lfedd001/renderman.html)
