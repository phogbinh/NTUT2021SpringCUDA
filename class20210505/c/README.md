# Environment
* Windows 10
* Visual Studio 2019
* OpenCV 4.2.0
* CUDA Runtime 10.2

# Preparation
## 1. Add OpenCV binary directory to environment variable `Path`
`D:\opencv420_vc14_vc15\build\x64\vc14\bin` (example)

## 2. Create environment variables
### Include directory
* Variable name: `OPENCV420_INCLUDE`
* Variable value: `D:\opencv420_vc14_vc15\build\include` (example)

### Library directory
* Variable name: `OPENCV420_VC14_LIB`
* Variable value: `D:\opencv420_vc14_vc15\build\x64\vc14\lib` (example)

## 3. Set solution's configuration and platform
* Solution configuration: `Debug`
* Solution platform: `x64`