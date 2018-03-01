//! This crate provides simple Iterator for enumerating ractangle.
//!
//! # Examples

extern crate euclid;
extern crate num_traits;
use std::ops::Range;
use euclid::TypedRect;
use num_traits::Num;
use num_traits::cast::ToPrimitive;
/// RectRange is
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RectRange<T: Num + PartialOrd> {
    x_range: Range<T>,
    y_range: Range<T>,
}

impl<T: Num + PartialOrd> RectRange<T> {
    pub fn new(lx: T, ly: T, ux: T, uy: T) -> Option<RectRange<T>> {
        RectRange::from_ranges(lx..ux, ly..uy)
    }
    pub fn zero_start(x: T, y: T) -> Option<RectRange<T>> {
        RectRange::from_ranges(T::zero()..x, T::zero()..y)
    }
    pub fn from_ranges(x: Range<T>, y: Range<T>) -> Option<RectRange<T>> {
        if !Self::range_ok(&x) || !Self::range_ok(&y) {
            return None;
        }
        Some(RectRange {
            x_range: x,
            y_range: y,
        })
    }
    fn range_ok(r: &Range<T>) -> bool {
        r.start < r.end
    }
}

impl<T: Num + PartialOrd + Clone> RectRange<T> {
    pub fn x(&self) -> Range<T> {
        self.x_range.clone()
    }
    pub fn y(&self) -> Range<T> {
        self.y_range.clone()
    }
    pub fn slide(self, t: (T, T)) -> RectRange<T> {
        RectRange {
            x_range: self.x_range.start + t.0.clone()..self.x_range.end + t.0,
            y_range: self.y_range.start + t.1.clone()..self.y_range.end + t.1,
        }
    }
}

impl<T: Num + PartialOrd + Copy> RectRange<T> {
    pub fn from_rect<U>(rect: TypedRect<T, U>) -> Option<RectRange<T>> {
        let orig_x = rect.origin.x;
        let orig_y = rect.origin.y;
        RectRange::from_ranges(
            orig_x..orig_x + rect.size.width,
            orig_y..orig_y + rect.size.height,
        )
    }
}

impl<T: Num + PartialOrd + Copy> IntoIterator for RectRange<T> {
    type Item = (T, T);
    type IntoIter = RectIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        RectIter {
            x: self.x_range.start,
            y: self.y_range.start,
            range: self,
        }
    }
}

pub struct RectIter<T: Num + PartialOrd + Copy> {
    x: T,
    y: T,
    range: RectRange<T>,
}

impl<T: Num + PartialOrd + Copy> Iterator for RectIter<T> {
    type Item = (T, T);
    fn next(&mut self) -> Option<(T, T)> {
        if self.y >= self.range.y_range.end {
            return None;
        }
        let before = (self.x, self.y);
        let nxt_x = T::one() + self.x;
        if nxt_x < self.range.x_range.end {
            self.x = nxt_x;
        } else {
            self.x = self.range.x_range.start;
            self.y = T::one() + self.y;
        }
        Some(before)
    }
}

pub trait XyGet {
    type Item;
    fn xy_get<T: ToPrimitive>(&self, x: T, y: T) -> Option<&Self::Item>;
}

pub trait TupleGet: XyGet {
    fn tuple_get<T: ToPrimitive>(&self, t: (T, T)) -> Option<&Self::Item> {
        self.xy_get(t.0, t.1)
    }
}

pub trait XyGetMut {
    type Item;
    fn xy_get_mut<T: ToPrimitive>(&mut self, x: T, y: T) -> Option<&mut Self::Item>;
}

pub trait TupleGetMut: XyGetMut {
    fn tuple_get_mut<T: ToPrimitive>(&mut self, t: (T, T)) -> Option<&mut Self::Item> {
        self.xy_get_mut(t.0, t.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn iter_test_normal() {
        let r = RectRange::from_ranges(4..7, 3..5).unwrap();
        let correct = [(4, 3), (5, 3), (6, 3), (4, 4), (5, 4), (6, 4)];
        for (i, (x, y)) in r.into_iter().enumerate() {
            assert_eq!(correct[i], (x, y));
        }
    }
}
