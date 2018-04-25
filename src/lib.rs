//! This crate provides simple Iterator for enumerating rectangle.
#![feature(iterator_try_fold)]

#[cfg(feature = "euclid")]
extern crate euclid;

#[cfg(feature = "image")]
#[allow(unused_imports)]
extern crate image;

extern crate num_traits;
extern crate tuple_map;

#[cfg(feature = "serde")]
#[allow(unused_imports)]
#[macro_use]
extern crate serde;

#[allow(unused_imports)]
use std::ops::{Deref, DerefMut, Range};

#[cfg(feature = "euclid")]
use euclid::{rect, TypedPoint2D, TypedRect, TypedVector2D};
use num_traits::cast::{FromPrimitive, ToPrimitive};
use num_traits::Num;
use tuple_map::TupleMap2;

#[cfg(feature = "image")]
use image::{ImageBuffer, Pixel};
use std::error::Error;
use std::fmt;

/// Error type for invalid access to 2D array.
#[derive(Copy, Clone, Debug, Default)]
pub struct IndexError {
    pub x: i64,
    pub y: i64,
}

unsafe impl Send for IndexError {}
unsafe impl Sync for IndexError {}

impl IndexError {
    fn new<T: ToPrimitive>(x: T, y: T) -> IndexError {
        let (x, y) = (x, y).map(|i| i.to_i64().unwrap());
        IndexError { x: x, y: y }
    }
}

impl Error for IndexError {
    fn description(&self) -> &str {
        "Invalid Index access"
    }
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Index Error at x: {}, y: {}", self.x, self.y)
    }
}

/// To manipulate many Point libraries in the same way, we use tuple as entry points of API.
/// If you implement IntoTuple2 to your point type, you can use it as Point in this library.
/// # Example
/// ```
/// # extern crate rect_iter; fn main() {
/// use rect_iter::IntoTuple2;
/// struct Point {
///     x: f64,
///     y: f64,
/// }
/// impl IntoTuple2<f64> for Point {
///     fn into_tuple2(self) -> (f64, f64) {
///         (self.x, self.y)
///     }
/// }
/// # }
/// ```
pub trait IntoTuple2<T> {
    fn into_tuple2(self) -> (T, T);
}

/// To manipulate many Point libraries in the same way, we use tuple as entry points of API.
/// # Example
/// ```
/// # extern crate rect_iter; fn main() {
/// use rect_iter::FromTuple2;
/// struct Point {
///     x: f64,
///     y: f64,
/// }
/// impl FromTuple2<f64> for Point {
///     fn from_tuple2(t: (f64, f64)) -> Self {
///         Point { x: t.0, y: t.1 }
///     }
/// }
/// # }
/// ```
pub trait FromTuple2<T> {
    fn from_tuple2(tuple: (T, T)) -> Self;
}

#[cfg(feature = "euclid")]
impl<T: Clone, U> IntoTuple2<T> for TypedPoint2D<T, U> {
    fn into_tuple2(self) -> (T, T) {
        (self.x, self.y)
    }
}

#[cfg(feature = "euclid")]
impl<T: Clone, U> IntoTuple2<T> for TypedVector2D<T, U> {
    fn into_tuple2(self) -> (T, T) {
        (self.x, self.y)
    }
}

impl<T: Clone> IntoTuple2<T> for (T, T) {
    fn into_tuple2(self) -> (T, T) {
        self.clone()
    }
}

/// RectRange is rectangle representation using `std::ops::Range`.
///
/// Diffrent from Range<T>, RectRange itself isn't a iterator, but
/// has `IntoIterator` implementation and `iter` method(with 'clone').
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
    pub fn from_point<P: IntoTuple2<T>>(p: P) -> Option<RectRange<T>> {
        let p = p.into_tuple2();
        RectRange::from_ranges(T::zero()..p.0, T::zero()..p.1)
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
    /// get x range
    pub fn cloned_x(&self) -> Range<T> {
        self.x_range.clone()
    }
    /// get y range
    pub fn cloned_y(&self) -> Range<T> {
        self.y_range.clone()
    }
    /// slide range by the given point
    pub fn slide<P: IntoTuple2<T>>(self, t: P) -> RectRange<T> {
        let t = t.into_tuple2();
        RectRange {
            x_range: self.x_range.start + t.0.clone()..self.x_range.end + t.0,
            y_range: self.y_range.start + t.1.clone()..self.y_range.end + t.1,
        }
    }
    /// slide start point without checking
    pub fn slide_start<P: IntoTuple2<T>>(self, t: P) -> RectRange<T> {
        let t = t.into_tuple2();
        RectRange {
            x_range: self.x_range.start + t.0.clone()..self.x_range.end,
            y_range: self.y_range.start + t.1.clone()..self.y_range.end,
        }
    }
    /// slide end point without checking
    pub fn slide_end<P: IntoTuple2<T>>(self, t: P) -> RectRange<T> {
        let t = t.into_tuple2();
        RectRange {
            x_range: self.x_range.start..self.x_range.end + t.0,
            y_range: self.y_range.start..self.y_range.end + t.1,
        }
    }
    /// the length in the x-axis deirection
    pub fn xlen(&self) -> T {
        let r = self.x_range.clone();
        r.end - r.start
    }
    /// the length of in the y-axis deirection
    pub fn ylen(&self) -> T {
        let r = self.y_range.clone();
        r.end - r.start
    }
    /// judges if 2 ranges have intersection
    pub fn intersects(&self, other: &RectRange<T>) -> bool {
        let not_inter = |r1: &Range<T>, r2: &Range<T>| r1.end <= r2.start || r2.end <= r1.start;
        !(not_inter(&self.x_range, &other.x_range) || not_inter(&self.y_range, &other.y_range))
    }
    /// gets the intersection of 2 ranges
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
    #[cfg(feature = "euclid")]
    pub fn from_rect<U>(rect: TypedRect<T, U>) -> Option<RectRange<T>> {
        let orig_x = rect.origin.x;
        let orig_y = rect.origin.y;
        RectRange::from_ranges(
            orig_x..orig_x + rect.size.width,
            orig_y..orig_y + rect.size.height,
        )
    }
    #[cfg(feature = "euclid")]
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
    pub fn from_corners<P: IntoTuple2<T>>(lu: P, rd: P) -> Option<RectRange<T>> {
        let lu = lu.into_tuple2();
        let rd = rd.into_tuple2();
        RectRange::new(lu.0, lu.1, rd.0, rd.1)
    }
    pub fn iter(&self) -> RectIter<T> {
        self.clone().into_iter()
    }
    pub fn scale(self, sc: T) -> RectRange<T> {
        let scale_impl = |r: Range<T>, s| r.start * s..r.end * s;
        RectRange {
            x_range: scale_impl(self.x_range, sc),
            y_range: scale_impl(self.y_range, sc),
        }
    }
}

macro_rules! __cast_impl {
    ($method:ident, $x:expr, $y:expr) => {
        Some(RectRange {
            x_range: $x.start.$method()?..$x.end.$method()?,
            y_range: $y.start.$method()?..$y.end.$method()?,
        })
    };
}
impl<T: Num + PartialOrd + ToPrimitive + Copy> RectRange<T> {
    pub fn to_u8(self) -> Option<RectRange<u8>> {
        __cast_impl!(to_u8, self.x_range, self.y_range)
    }
    pub fn to_u16(self) -> Option<RectRange<u16>> {
        __cast_impl!(to_u16, self.x_range, self.y_range)
    }
    pub fn to_u32(self) -> Option<RectRange<u32>> {
        __cast_impl!(to_u32, self.x_range, self.y_range)
    }
    pub fn to_u64(self) -> Option<RectRange<u64>> {
        __cast_impl!(to_u64, self.x_range, self.y_range)
    }
    pub fn to_i8(self) -> Option<RectRange<i8>> {
        __cast_impl!(to_i8, self.x_range, self.y_range)
    }
    pub fn to_i16(self) -> Option<RectRange<i16>> {
        __cast_impl!(to_i16, self.x_range, self.y_range)
    }
    pub fn to_i32(self) -> Option<RectRange<i32>> {
        __cast_impl!(to_i32, self.x_range, self.y_range)
    }
    pub fn to_i64(self) -> Option<RectRange<i64>> {
        __cast_impl!(to_i64, self.x_range, self.y_range)
    }
    pub fn to_usize(self) -> Option<RectRange<usize>> {
        __cast_impl!(to_usize, self.x_range, self.y_range)
    }
}

impl<T: Num + PartialOrd + Copy + FromPrimitive + ToPrimitive> RectRange<T> {
    /// returns `self.xlen * self.ylen` as usize
    pub fn len(&self) -> usize {
        let (width, height) = (&self.x_range, &self.y_range)
            .map(|r| r.clone())
            .map(|r| r.end - r.start);
        (width * height)
            .to_usize()
            .expect("[RectRange::len] invalid cast")
    }
    /// return 'nth' element as iterator
    pub fn nth(&self, n: usize) -> Option<(T, T)> {
        let width = self.x_range.end - self.x_range.start;
        let width = width.to_usize()?;
        let x = T::from_usize(n % width)? + self.x_range.start;
        let y = T::from_usize(n / width)? + self.y_range.start;
        if y >= self.y_range.end {
            None
        } else {
            Some((x, y))
        }
    }
}

impl<T: Num + PartialOrd + Copy> IntoIterator for RectRange<T> {
    type Item = (T, T);
    type IntoIter = RectIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        RectIter {
            front: (self.x_range.start, self.y_range.start),
            back: (self.x_range.end - T::one(), self.y_range.end - T::one()),
            end: false,
            x_range: self.x_range.clone(),
            y_range: self.y_range.clone(),
        }
    }
}

/// An Iterator type enumerating rectangle area.
///
/// You can construct it by RectRange.
/// # Example
/// ```
/// # extern crate rect_iter; fn main() {
/// use rect_iter::RectRange;
/// let range = RectRange::zero_start(3, 4);
/// for point in range {
///     // some code here...
/// }
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct RectIter<T: Num + PartialOrd + Copy> {
    front: (T, T),
    back: (T, T),
    end: bool,
    x_range: Range<T>,
    y_range: Range<T>,
}

impl<T: Num + PartialOrd + Copy> Iterator for RectIter<T> {
    type Item = (T, T);
    fn next(&mut self) -> Option<Self::Item> {
        if self.end {
            return None;
        }
        if self.front >= self.back {
            self.end = true;
            return Some(self.front);
        }
        let before = self.front;
        if self.front.0 == self.x_range.end - T::one() {
            self.front.0 = self.x_range.start;
            self.front.1 = T::one() + self.front.1;
        } else {
            self.front.0 = T::one() + self.front.0;
        }
        Some(before)
    }
}

impl<T: Num + PartialOrd + Copy> DoubleEndedIterator for RectIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.end {
            return None;
        }
        if self.front >= self.back {
            self.end = true;
            return Some(self.front);
        }
        let before = self.back;
        if self.back.0 == self.x_range.start {
            self.back.0 = self.x_range.end - T::one();
            self.back.1 = self.back.1 - T::one();
        } else {
            self.back.0 = self.back.0 - T::one();
        }
        Some(before)
    }
}

impl<T: Num + PartialOrd + Copy + ToPrimitive> ExactSizeIterator for RectIter<T> {
    fn len(&self) -> usize {
        if self.end {
            0
        } else {
            let (xlen, ylen) = self.back.sub(self.front).add((T::one(), T::one()));
            (xlen * ylen)
                .to_usize()
                .expect("[RectIter::len] invalid cast")
        }
    }
}

/// A trait which provides common access interfaces to 2D Array type.
pub trait Get2D {
    type Item;
    fn get_xy<T: ToPrimitive>(&self, x: T, y: T) -> Option<&Self::Item>;
    fn get_point<T: ToPrimitive, P: IntoTuple2<T>>(&self, t: P) -> Option<&Self::Item> {
        let t = t.into_tuple2();
        self.get_xy(t.0, t.1)
    }
    fn get_xy_r<T: ToPrimitive + Clone>(&self, x: T, y: T) -> Result<&Self::Item, IndexError> {
        let r = self.get_xy(x.clone(), y.clone());
        match r {
            Some(p) => Ok(p),
            None => Err(IndexError::new(x, y)),
        }
    }
    fn get_point_r<T: ToPrimitive + Clone, P: IntoTuple2<T>>(
        &self,
        t: P,
    ) -> Result<&Self::Item, IndexError> {
        let t = t.into_tuple2();
        self.get_xy_r(t.0, t.1)
    }
}

pub trait GetMut2D {
    type Item;
    fn get_mut_xy<T: ToPrimitive>(&mut self, x: T, y: T) -> Option<&mut Self::Item>;
    fn get_mut_point<T: ToPrimitive, P: IntoTuple2<T>>(&mut self, t: P) -> Option<&mut Self::Item> {
        let t = t.into_tuple2();
        self.get_mut_xy(t.0, t.1)
    }
    fn get_mut_xy_r<T: ToPrimitive + Clone>(
        &mut self,
        x: T,
        y: T,
    ) -> Result<&mut Self::Item, IndexError> {
        let r = self.get_mut_xy(x.clone(), y.clone());
        match r {
            Some(p) => Ok(p),
            None => Err(IndexError::new(x, y)),
        }
    }
    fn get_mut_point_r<T: ToPrimitive + Clone, P: IntoTuple2<T>>(
        &mut self,
        t: P,
    ) -> Result<&mut Self::Item, IndexError> {
        let t = t.into_tuple2();
        self.get_mut_xy_r(t.0, t.1)
    }
}

impl<D> Get2D for Vec<Vec<D>> {
    type Item = D;
    fn get_xy<T: ToPrimitive>(&self, x: T, y: T) -> Option<&Self::Item> {
        Some(self.get(y.to_usize()?)?.get(x.to_usize()?)?)
    }
}

impl<D> GetMut2D for Vec<Vec<D>> {
    type Item = D;
    fn get_mut_xy<T: ToPrimitive>(&mut self, x: T, y: T) -> Option<&mut Self::Item> {
        Some(self.get_mut(y.to_usize()?)?.get_mut(x.to_usize()?)?)
    }
}

pub fn copy_rect_conv<T, U, I, J>(
    source: &impl Get2D<Item = T>,
    dest: &mut impl GetMut2D<Item = U>,
    source_range: RectRange<I>,
    dest_range: RectRange<J>,
    convert: impl Fn(&T) -> U,
) -> Result<(), IndexError>
where
    I: Num + PartialOrd + ToPrimitive + Copy,
    J: Num + PartialOrd + ToPrimitive + Copy,
{
    source_range
        .into_iter()
        .zip(dest_range.into_iter())
        .try_for_each(|(s, d)| {
            *dest.get_mut_point_r(d)? = convert(source.get_point_r(s)?);
            Ok(())
        })
}

pub fn copy_rect<T, I, J>(
    source: &impl Get2D<Item = T>,
    dest: &mut impl GetMut2D<Item = T>,
    source_range: RectRange<I>,
    dest_range: RectRange<J>,
) -> Result<(), IndexError>
where
    T: Clone,
    I: Num + PartialOrd + ToPrimitive + Copy,
    J: Num + PartialOrd + ToPrimitive + Copy,
{
    source_range
        .into_iter()
        .zip(dest_range.into_iter())
        .try_for_each(|(s, d)| {
            *dest.get_mut_point_r(d)? = source.get_point_r(s)?.clone();
            Ok(())
        })
}

pub fn gen_rect_conv<D, T, U, I, J>(
    source: &impl Get2D<Item = T>,
    gen_dist: impl Fn() -> D,
    source_range: RectRange<I>,
    dest_range: RectRange<J>,
    convert: impl Fn(&T) -> U,
) -> Result<D, IndexError>
where
    D: GetMut2D<Item = U> + Default,
    T: Clone,
    I: Num + PartialOrd + ToPrimitive + Copy,
    J: Num + PartialOrd + ToPrimitive + Copy,
{
    source_range
        .into_iter()
        .zip(dest_range.into_iter())
        .try_fold(gen_dist(), |mut dest, (s, d)| {
            *dest.get_mut_point_r(d)? = convert(source.get_point_r(s)?);
            Ok(dest)
        })
}

pub fn gen_rect<D, T, I, J>(
    source: &impl Get2D<Item = T>,
    gen_dist: impl Fn() -> D,
    source_range: RectRange<I>,
    dest_range: RectRange<J>,
) -> Result<D, IndexError>
where
    D: GetMut2D<Item = T> + Default,
    T: Clone,
    I: Num + PartialOrd + ToPrimitive + Copy,
    J: Num + PartialOrd + ToPrimitive + Copy,
{
    source_range
        .into_iter()
        .zip(dest_range.into_iter())
        .try_fold(gen_dist(), |mut dest, (s, d)| {
            *dest.get_mut_point_r(d)? = source.get_point_r(s)?.clone();
            Ok(dest)
        })
}

#[cfg(feature = "image")]
impl<P, C> Get2D for ImageBuffer<P, C>
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
    C: Deref<Target = [P::Subpixel]>,
{
    type Item = P;
    fn get_xy<T: ToPrimitive>(&self, x: T, y: T) -> Option<&Self::Item> {
        let (x, y) = (x.to_u32()?, y.to_u32()?);
        if x >= self.width() || y >= self.height() {
            None
        } else {
            Some(self.get_pixel(x, y))
        }
    }
}

#[cfg(feature = "image")]
impl<P, C> GetMut2D for ImageBuffer<P, C>
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
    C: Deref<Target = [P::Subpixel]> + DerefMut,
{
    type Item = P;
    fn get_mut_xy<T: ToPrimitive>(&mut self, x: T, y: T) -> Option<&mut Self::Item> {
        let (x, y) = (x.to_u32()?, y.to_u32()?);
        if x >= self.width() || y >= self.height() {
            None
        } else {
            Some(self.get_pixel_mut(x, y))
        }
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
    fn iter_test_rev() {
        let r = RectRange::from_ranges(4..7, 3..5).unwrap();
        let correct = [(4, 3), (5, 3), (6, 3), (4, 4), (5, 4), (6, 4)];
        for (&c, t) in correct.into_iter().rev().zip(r.into_iter().rev()) {
            assert_eq!(c, t);
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
    #[test]
    fn test_get_vec() {
        let a = vec![vec![3; 5]; 7];
        assert_eq!(Some(&3), a.get_xy(3, 3));
        assert_eq!(None, a.get_xy(5, 7));
    }
    #[test]
    fn test_copy_rect() {
        let mut a = vec![vec![3; 5]; 7];
        let r1 = RectRange::from_ranges(3..5, 2..6).unwrap();
        let b = vec![vec![80; 100]; 100];
        copy_rect(&b, &mut a, r1.clone(), r1.clone()).unwrap();
        r1.into_iter()
            .for_each(|p| assert_eq!(a.get_point(p), Some(&80)));
    }
    #[test]
    fn test_gen_rect() {
        let r = RectRange::zero_start(5, 7).unwrap();
        let b = vec![vec![80; 100]; 100];
        let a = gen_rect(&b, || vec![vec![0; 5]; 7], r.clone(), r.clone()).unwrap();
        for i in r {
            println!("{:?}", i);
        }
        assert_eq!(vec![vec![80; 5]; 7], a);
    }
    #[test]
    fn test_length() {
        let r = RectRange::zero_start(7, 8).unwrap();
        assert_eq!(r.len(), 56);
        let iter = r.into_iter();
        assert_eq!(iter.len(), 56);
    }
    #[test]
    fn test_nth() {
        let r = RectRange::from_ranges(4..7, 3..7).unwrap();
        assert_eq!(r.nth(7), r.iter().nth(7));
        assert_eq!(r.nth(12), None);
    }
}
