package com.github.jbellis.jvector.annotations;

import java.lang.annotation.*;

/**
 * Type uses marked Unshared indicate an Object that will not be reused across returning method invocations.
 * Counterpart of Shared.
 */
@Documented
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE_USE)
public @interface Unshared {
}
