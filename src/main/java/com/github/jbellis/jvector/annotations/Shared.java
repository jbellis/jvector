package com.github.jbellis.jvector.annotations;

import java.lang.annotation.*;

/**
 * Type uses marked Shared indicate the Object may be reused across returning method invocations.
 * Make a deep copy if you want to use it across calls.
 */
@Documented
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE_USE) // TYPE_USE used instead of METHOD as the annotation travels better in some tooling
public @interface Shared {
}
