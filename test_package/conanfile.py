from conans import ConanFile, CMake
import os


channel = os.getenv("CONAN_CHANNEL", "stable")
username = os.getenv("CONAN_USERNAME", "bitprim")
version = "0.5"


class BitprimcoreTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = "bitprim-core/%s@%s/%s" % (version, username, channel)
    generators = "cmake"
    # build_policy = "missing"

    def build(self):
        cmake = CMake(self)
        cmake.definitions["CMAKE_VERBOSE_MAKEFILE"] = "ON"
        # Current dir is "test_package/build/<build_id>" and CMakeLists.txt is in "test_package"
        cmake.configure(source_dir=self.source_folder, build_dir="./")
        cmake.build()

    def imports(self):
        self.copy("*.dll", dst="bin", src="bin")
        self.copy("*.dylib*", dst="bin", src="lib")

    def test(self):
        os.chdir("bin")
        self.run(".%sexample" % os.sep)
