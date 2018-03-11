# rect-iter
[![Build Status](https://travis-ci.org/kngwyu/rect-iter.svg?branch=master)](https://travis-ci.org/kngwyu/rect-iter)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This library provides general iterator for enumerating Rectangle.

There are many libralies handle 2D rectangle area, so I think it's convinient if we can use those libraries in the same way.

# Example

with `image` feature:

``` rust
extern crate rect-iter;
extern crate image;
use image::DynamicImage;
use rect_iter::{gen_rect, RectRange};
fn main() {
    let img = image::open("a.png");
    let img = match img {
        DynamicImage::ImageRgba8(img) => img,
        x => x.to_rgba(),
    };
    let (x, y) = (img.width(), img.height());
    let img_range = RectRange::zero_start(x, y).unwrap();
    let red = gen_rect(&b, || vec![vec![0; x]; y], img_range.clone(), img_range.clone());
}

```

