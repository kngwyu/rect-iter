# rect-iter
[![Build Status](https://travis-ci.org/kngwyu/rect-iter.svg?branch=master)](https://travis-ci.org/kngwyu/rect-iter)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This library provides general iterator for enumerating Rectangle.

There are many libralies handle 2D rectangle area, so I think it's convinient if we can use those libraries in the same way.

# Example

with `image` feature:

``` rust
extern crate rect_iter;
extern crate euclid;
use euclid::TypedVector2D;
use rect_iter::{RectRange, FromTuple2, GetMut2D};
type MyVec = TypedVector2D<u64, ()>;
fn main() {
    let range = RectRange::from_ranges(4..9, 5..10).unwrap();
    let mut buffer = vec![vec![0.0; 100]; 100];
    range.iter().for_each(|t| {
        let len = MyVec::from_tuple2(t).to_f64().length();
        *buffer.get_mut_p(t) = len;
    });
}
```

