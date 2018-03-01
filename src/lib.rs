//! This crate provides simple Iterator for enumerating ractangle.
//!
//! # Examples

extern crate euclid;
extern crate num_traits;
#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;
use std::ops::Range;
use euclid::{rect, TypedPoint2D, TypedRect, TypedVector2D};
use num_traits::Num;
use num_traits::cast::ToPrimitive;

pub trait ToPoint<T> {
    fn to_point(&self) -> (T, T);
}

impl<T: Clone, U> ToPoint<T> for TypedPoint2D<T, U> {
    fn to_point(&self) -> (T, T) {
        (self.x.clone(), self.y.clone())
    }
}

impl<T: Clone, U> ToPoint<T> for TypedVector2D<T, U> {
    fn to_point(&self) -> (T, T) {
        (self.x.clone(), self.y.clone())
    }
}

impl<T: Clone> ToPoint<T> for (T, T) {
    fn to_point(&self) -> (T, T) {
        self.clone()
    }
}

/// RectRange is rectangle representation using `std::ops::Range`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    pub fn get_x(&self) -> &Range<T> {
        &self.x_range
    }
    pub fn get_y(&self) -> &Range<T> {
        &self.y_range
    }
    fn range_ok(r: &Range<T>) -> bool {
        r.start < r.end
    }
}

impl<T: Num + PartialOrd + Clone> RectRange<T> {
    pub fn cloned_x(&self) -> Range<T> {
        self.x_range.clone()
    }
    pub fn cloned_y(&self) -> Range<T> {
        self.y_range.clone()
    }
    pub fn slide(self, t: (T, T)) -> RectRange<T> {
        RectRange {
            x_range: self.x_range.start + t.0.clone()..self.x_range.end + t.0,
            y_range: self.y_range.start + t.1.clone()..self.y_range.end + t.1,
        }
    }
    pub fn xlen(&self) -> T {
        let r = self.x_range.clone();
        r.end - r.start
    }
    pub fn ylen(&self) -> T {
        let r = self.y_range.clone();
        r.end - r.start
    }
    pub fn intersects(&self, other: &RectRange<T>) -> bool {
        let not_inter = |r1: &Range<T>, r2: &Range<T>| r1.end <= r2.start || r2.end <= r1.start;
        !(not_inter(&self.x_range, &other.x_range) || not_inter(&self.y_range, &other.y_range))
    }
    pub fn intersection(&self, other: &RectRange<T>) -> Option<RectRange<T>> {
        let inter = |r1: Range<T>, r2: Range<T>| {
            let s = max(r1.start, r2.start);
            let e = min(r1.end, r2.end);
            if s >= e {
                None
            } else {
                Some(s..e)
            }
        };
        Some(RectRange {
            x_range: inter(self.x_range.clone(), other.x_range.clone())?,
            y_range: inter(self.y_range.clone(), other.y_range.clone())?,
        })
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
    pub fn to_rect<U>(&self) -> TypedRect<T, U> {
        let orig_x = self.x_range.start;
        let orig_y = self.y_range.start;
        rect(
            orig_x,
            orig_y,
            self.x_range.end - orig_x,
            self.y_range.end - orig_y,
        )
    }
    pub fn from_corners<P: ToPoint<T>>(lu: P, rd: P) -> Option<RectRange<T>> {
        let lu = lu.to_point();
        let rd = rd.to_point();
        RectRange::new(lu.0, lu.1, rd.0, rd.1)
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

fn min<T: Clone + PartialOrd>(x: T, y: T) -> T {
    if x <= y {
        x
    } else {
        y
    }
}

fn max<T: Clone + PartialOrd>(x: T, y: T) -> T {
    if x >= y {
        x
    } else {
        y
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
    #[test]
    fn test_intersects_true() {
        let r1 = RectRange::from_ranges(4..7, 3..5).unwrap();
        let r2 = RectRange::from_ranges(6..10, 4..6).unwrap();
        assert_eq!(r1.intersects(&r2), true)
    }
    #[test]
    fn test_intersection_some() {
        let r1 = RectRange::from_ranges(4..7, 3..5).unwrap();
        let r2 = RectRange::from_ranges(6..10, 4..6).unwrap();
        let inter = RectRange::from_ranges(6..7, 4..5).unwrap();
        assert_eq!(r1.intersection(&r2).unwrap(), inter);
    }
    #[test]
    fn test_intersects_false() {
        let r1 = RectRange::from_ranges(4..7, 3..5).unwrap();
        let r2 = RectRange::from_ranges(7..9, 5..6).unwrap();
        assert_eq!(r1.intersects(&r2), false)
    }
    #[test]
    fn test_intersection_none() {
        let r1 = RectRange::from_ranges(4..7, 3..5).unwrap();
        let r2 = RectRange::from_ranges(7..9, 5..6).unwrap();
        assert!(r1.intersection(&r2).is_none());
    }
}
