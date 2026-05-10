/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.datafusion.jni;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Loads the native DataFusion library via JNI. */
public class NativeLoader {

  private static final Logger LOG = LoggerFactory.getLogger(NativeLoader.class);
  private static final String NATIVE_LIB_NAME = "comet";

  private static final String libraryToLoad = System.mapLibraryName(NATIVE_LIB_NAME);
  private static boolean loaded = false;
  private static volatile Throwable loadErr = null;
  private static final String searchPattern = "libcomet-";

  static {
    try {
      load();
    } catch (Throwable th) {
      LOG.warn("Failed to load native library", th);
      System.err.println("Failed to load native library: " + th.getMessage());
      loadErr = th;
    }
  }

  public static synchronized boolean isLoaded() throws Throwable {
    if (loadErr != null) {
      throw loadErr;
    }
    return loaded;
  }

  static synchronized void load() {
    if (loaded) {
      return;
    }

    cleanupOldTempLibs();

    if (!checkArch()) {
      LOG.warn(
          "Native library disabled: JDK compiled for x86_64 on Apple Silicon. "
              + "Please install an ARM64 JDK.");
      return;
    }

    try {
      System.loadLibrary(NATIVE_LIB_NAME);
      loaded = true;
    } catch (UnsatisfiedLinkError ex) {
      bundleLoadLibrary();
    }

    String logPath = System.getProperty("datafusion.native.log.path", "");
    String logLevel = System.getenv("DATAFUSION_LOG_LEVEL");
    init(logPath, logLevel);
  }

  private static void bundleLoadLibrary() {
    String resourceName = resourceName();
    InputStream is = NativeLoader.class.getResourceAsStream(resourceName);
    if (is == null) {
      throw new UnsupportedOperationException(
          "Unsupported OS/arch, cannot find "
              + resourceName
              + ". Please try building from source.");
    }

    File tempLib = null;
    File tempLibLock = null;
    try {
      tempLibLock = File.createTempFile(searchPattern, "." + os().libExtension + ".lck");
      tempLib = new File(tempLibLock.getAbsolutePath().replaceFirst(".lck$", ""));
      Files.copy(is, tempLib.toPath(), StandardCopyOption.REPLACE_EXISTING);
      System.load(tempLib.getAbsolutePath());
      loaded = true;
    } catch (IOException e) {
      throw new IllegalStateException("Cannot unpack native library: " + e);
    } finally {
      if (!loaded) {
        if (tempLib != null && tempLib.exists()) {
          tempLib.delete();
        }
        if (tempLibLock != null && tempLibLock.exists()) {
          tempLibLock.delete();
        }
      } else {
        if (tempLib != null) {
          tempLib.deleteOnExit();
        }
        if (tempLibLock != null) {
          tempLibLock.deleteOnExit();
        }
      }
    }
  }

  private static void cleanupOldTempLibs() {
    String tempFolder = System.getProperty("java.io.tmpdir");
    File dir = new File(tempFolder);

    File[] tempLibFiles =
        dir.listFiles(
            new FilenameFilter() {
              public boolean accept(File dir, String name) {
                return name.startsWith(searchPattern) && !name.endsWith(".lck");
              }
            });

    if (tempLibFiles != null) {
      for (File tempLibFile : tempLibFiles) {
        File lckFile = new File(tempLibFile.getAbsolutePath() + ".lck");
        if (!lckFile.exists()) {
          try {
            tempLibFile.delete();
          } catch (SecurityException e) {
            LOG.debug("Failed to delete old temp lib: {}", tempLibFile, e);
          }
        }
      }
    }
  }

  private enum OS {
    WINDOWS("win32", "so"),
    LINUX("linux", "so"),
    MAC("darwin", "dylib"),
    SOLARIS("solaris", "so");

    public final String name;
    public final String libExtension;

    OS(String name, String libExtension) {
      this.name = name;
      this.libExtension = libExtension;
    }
  }

  private static String arch() {
    return System.getProperty("os.arch");
  }

  private static OS os() {
    String osName = System.getProperty("os.name");
    if (osName.contains("Linux")) {
      return OS.LINUX;
    } else if (osName.contains("Mac")) {
      return OS.MAC;
    } else if (osName.contains("Windows")) {
      return OS.WINDOWS;
    } else if (osName.contains("Solaris") || osName.contains("SunOS")) {
      return OS.SOLARIS;
    } else {
      throw new UnsupportedOperationException("Unsupported operating system: " + osName);
    }
  }

  private static boolean checkArch() {
    if (os() == OS.MAC) {
      try {
        String javaArch = arch();
        Process process = Runtime.getRuntime().exec("uname -a");
        if (process.waitFor() == 0) {
          java.io.BufferedReader in =
              new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()));
          String line;
          while ((line = in.readLine()) != null) {
            if (javaArch.equals("x86_64") && line.contains("ARM64")) {
              return false;
            }
          }
        }
      } catch (IOException | InterruptedException e) {
        LOG.warn("Error parsing host architecture", e);
      }
    }
    return true;
  }

  private static String resourceName() {
    OS os = os();
    return "/org/apache/datafusion/jni/" + os.name + "/" + arch() + "/" + libraryToLoad;
  }

  static native void init(String logConfPath, String logLevel);
}
