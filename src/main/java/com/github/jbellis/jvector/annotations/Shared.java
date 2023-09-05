package com.github.jbellis.jvector.annotations;

import java.lang.annotation.*;

/**
 * Type used marked Shared indicate the Object may be reused across returning method invocations.
 * Make a deep copy if you want to use it across calls.
 */
@Documented
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE_USE)
public @interface Shared {
}
