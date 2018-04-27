# rect-iter
[![Build Status](https://travis-ci.org/kngwyu/rect-iter.svg?branch=master)](https://travis-ci.org/kngwyu/rect-iter)
[![crate.io](http://meritbadge.herokuapp.com/rect-iter)](https://crates.io/crates/rect-iter)
[![Documentation](https://docs.rs/rect-iter/badge.svg)](https://docs.rs/rect-iter)

This library provides general iterator for enumerating Rectangle.

There are many libralies which handle 2D rectangle area, so I think it's convinient if we can use those libraries in the same way.

# Example

with `euclid` feature(it's included by default):

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

# License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.
