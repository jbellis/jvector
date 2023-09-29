/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.util;

/** Some useful constants. */
public final class Constants {
  private Constants() {} // can't construct

    /** The value of <code>System.getProperty("os.name")</code>. * */
  public static final String OS_NAME = System.getProperty("os.name");

    /** The value of <code>System.getProperty("os.arch")</code>. */
  public static final String OS_ARCH = System.getProperty("os.arch");

    /** True iff running on a 64bit JVM */
  public static final boolean JRE_IS_64BIT;

  static {
    boolean is64Bit = false;
    String datamodel = null;
    try {
      datamodel = System.getProperty("sun.arch.data.model");
      if (datamodel != null) {
        is64Bit = datamodel.contains("64");
      }
    } catch (
        @SuppressWarnings("unused")
        SecurityException ex) {
    }
    if (datamodel == null) {
      if (OS_ARCH != null && OS_ARCH.contains("64")) {
        is64Bit = true;
      } else {
        is64Bit = false;
      }
    }
    JRE_IS_64BIT = is64Bit;
  }
}
