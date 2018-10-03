//! This crate provides simple Iterator for enumerating rectangle.
//! # Example
//! ```rust
//! extern crate rect_iter;
//! extern crate euclid;
//! use euclid::TypedVector2D;
//! use rect_iter::{RectRange, FromTuple2, GetMut2D};
//! type MyVec = TypedVector2D<u64, ()>;
//! fn main() {
//!     let range = RectRange::from_ranges(4..9, 5..10).unwrap();
//!     let mut buffer = vec![vec![0.0; 100]; 100];
//!     range.iter().for_each(|t| {
//!         let len = MyVec::from_tuple2(t).to_f64().length();
//!         *buffer.get_mut_p(t) = len;
//!     });
//! }
//! ```

#[cfg(feature = "euclid")]
extern crate euclid;

#[cfg(feature = "image")]
#[allow(unused_imports)]
extern crate image;

#[cfg(feature = "ndarray")]
#[allow(unused_imports)]
extern crate ndarray;
extern crate num_traits;
extern crate tuple_map;

#[cfg(feature = "serde")]
#[allow(unused_imports)]
#[macro_use]
extern crate serde;

#[allow(unused_imports)]
use std::ops::{Deref, DerefMut, Range};

#[cfg(feature = "euclid")]
use euclid::{point2, rect, vec2, TypedPoint2D, TypedRect, TypedVector2D};

#[cfg(feature = "ndarray")]
use ndarray::{ArrayBase, Data, DataMut, Ix2};

use num_traits::cast::{FromPrimitive, ToPrimitive};
use num_traits::Num;
use tuple_map::TupleMap2;

#[cfg(feature = "image")]
use image::{ImageBuffer, Pixel};
use std::error::Error;
use std::fmt;

/// Error type for invalid access to 2D array.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum IndexError {
    X(i64),
    Y(i64),
}

unsafe impl Send for IndexError {}
unsafe impl Sync for IndexError {}

impl Error for IndexError {
    fn description(&self) -> &str {
        "Invalid Index access"
    }
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            IndexError::X(x) => write!(f, "Invalid Index x: {}", x),
            IndexError::Y(y) => write!(f, "Invalid Index y: {}", y),
        }
    }
}

/// To manipulate many Point libraries in the same way, we use tuple as a entry point of API.
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

impl<T> IntoTuple2<T> for (T, T) {
    fn into_tuple2(self) -> (T, T) {
        self
    }
}

#[cfg(feature = "euclid")]
impl<T: Copy, U> FromTuple2<T> for TypedPoint2D<T, U> {
    fn from_tuple2(t: (T, T)) -> TypedPoint2D<T, U> {
        point2(t.0, t.1)
    }
}

#[cfg(feature = "euclid")]
impl<T: Clone, U> FromTuple2<T> for TypedVector2D<T, U> {
    fn from_tuple2(t: (T, T)) -> TypedVector2D<T, U> {
        vec2(t.0, t.1)
    }
}

impl<T> FromTuple2<T> for (T, T) {
    fn from_tuple2(t: (T, T)) -> Self {
        t
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
    /// construct a range from left x, lower y, right x, upper y
    pub fn new(lx: T, ly: T, ux: T, uy: T) -> Option<RectRange<T>> {
        RectRange::from_ranges(lx..ux, ly..uy)
    }
    /// construct a range `x_range: 0..x, y_range: 0..y`
    pub fn zero_start(x: T, y: T) -> Option<RectRange<T>> {
        RectRange::from_ranges(T::zero()..x, T::zero()..y)
    }
    /// construct a range `x_range: 0..p.x, y_range: 0..p.y`
    pub fn from_point<P: IntoTuple2<T>>(p: P) -> Option<RectRange<T>> {
        let p = p.into_tuple2();
        RectRange::from_ranges(T::zero()..p.0, T::zero()..p.1)
    }
    /// construct a range `x_range: 0..x, y_range: 0..y`
    pub fn from_ranges(x_range: Range<T>, y_range: Range<T>) -> Option<RectRange<T>> {
        if !Self::range_ok(&x_range) || !Self::range_ok(&y_range) {
            return None;
        }
        Some(RectRange { x_range, y_range })
    }
    /// checks if the range contains the point
    pub fn contains<P: IntoTuple2<T>>(&self, p: P) -> bool {
        let (x, y) = p.into_tuple2();
        Self::contains_(&self.x_range, x) && Self::contains_(&self.y_range, y)
    }
    /// get a reference of x range
    pub fn get_x(&self) -> &Range<T> {
        &self.x_range
    }
    /// get a reference of y range
    pub fn get_y(&self) -> &Range<T> {
        &self.y_range
    }
    /// get a mutable reference of x range
    pub fn get_mut_x(&mut self) -> &mut Range<T> {
        &mut self.x_range
    }
    /// get a mutable reference of y range
    pub fn get_mut_y(&mut self) -> &mut Range<T> {
        &mut self.y_range
    }
    /// checks if the range is valid or not
    pub fn is_valid(&self) -> bool {
        Self::range_ok(&self.x_range) && Self::range_ok(&self.y_range)
    }
    fn range_ok(r: &Range<T>) -> bool {
        r.start < r.end
    }
    /// we should switch to `RangeBound::contains` if it becomes stable
    fn contains_(r: &Range<T>, val: T) -> bool {
        r.start <= val && val < r.end
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
    /// calc the area of the range
    pub fn area(&self) -> T {
        self.xlen() * self.ylen()
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
    /// get the upper left corner(inclusive)
    pub fn upper_left(&self) -> (T, T) {
        (&self.x_range, &self.y_range).map(|r| r.start.clone())
    }
    /// get the upper right corner(inclusive)
    pub fn upper_right(&self) -> (T, T) {
        let x = self.x_range.end.clone() - T::one();
        let y = self.y_range.start.clone();
        (x, y)
    }
    /// get the lower left corner(inclusive)
    pub fn lower_left(&self) -> (T, T) {
        let x = self.x_range.start.clone();
        let y = self.y_range.end.clone() - T::one();
        (x, y)
    }
    /// get the lower right corner(inclusive)
    pub fn lower_right(&self) -> (T, T) {
        (&self.x_range, &self.y_range).map(|r| r.end.clone() - T::one())
    }
    /// checks if the point is on the edge of the rectangle
    pub fn is_edge<P: IntoTuple2<T>>(&self, p: P) -> bool {
        let (x, y) = p.into_tuple2();
        self.contains((x.clone(), y.clone()))
            && ((x, self.x_range.clone()), (y, self.y_range.clone()))
                .any(|(p, r)| p == r.start || p == r.end - T::one())
    }
    /// checks if the point is on the vertical edge of the rectangle
    pub fn is_vert_edge<P: IntoTuple2<T>>(&self, p: P) -> bool {
        let (x, y) = p.into_tuple2();
        let range = self.x_range.clone();
        self.contains((x.clone(), y.clone())) && (x == range.start || x == range.end - T::one())
    }
    /// checks if the point is on the horizoni edge of the rectangle
    pub fn is_horiz_edge<P: IntoTuple2<T>>(&self, p: P) -> bool {
        let (x, y) = p.into_tuple2();
        let range = self.y_range.clone();
        self.contains((x.clone(), y.clone())) && (y == range.start || y == range.end - T::one())
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
    /// generate RectRange from corners(lower: inclusive, upper: exclusive)
    pub fn from_corners<P: IntoTuple2<T>>(lu: P, rd: P) -> Option<RectRange<T>> {
        let lu = lu.into_tuple2();
        let rd = rd.into_tuple2();
        RectRange::new(lu.0, lu.1, rd.0, rd.1)
    }
    /// get iterator from reference
    pub fn iter(&self) -> RectIter<T> {
        self.clone().into_iter()
    }
    /// expand the rectangle
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
    /// return 'nth' element
    /// same as RectIter::nth, but much faster(O(1))
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
    /// take Point and return 0-start unique index using the same order as RectIter
    pub fn index<P: IntoTuple2<T>>(&self, p: P) -> Option<usize> {
        let t = p.into_tuple2();
        if !self.contains(t) {
            return None;
        }
        let (x, y) = t.map(|i| i.to_usize());
        let (start_x, start_y) = (&self.x_range, &self.y_range).map(|r| r.start.to_usize());
        let xlen = self.xlen().to_usize()?;
        Some(x? - start_x? + (y? - start_y?) * xlen)
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
    fn get_xy<T: ToPrimitive>(&self, x: T, y: T) -> &Self::Item {
        self.try_get_xy(x, y).expect("[Get2d::get] Invalid index")
    }
    fn get_p<T: ToPrimitive, P: IntoTuple2<T>>(&self, p: P) -> &Self::Item {
        self.try_get_p(p).expect("[Get2d::get_p] Invalid index")
    }
    fn try_get_xy<T: ToPrimitive>(&self, x: T, y: T) -> Result<&Self::Item, IndexError>;
    fn try_get_p<T: ToPrimitive, P: IntoTuple2<T>>(&self, p: P) -> Result<&Self::Item, IndexError> {
        let t = p.into_tuple2();
        self.try_get_xy(t.0, t.1)
    }
}

pub trait GetMut2D: Get2D {
    fn get_mut_xy<T: ToPrimitive>(&mut self, x: T, y: T) -> &mut Self::Item {
        self.try_get_mut_xy(x, y)
            .expect("[Get2d::get] Invalid index")
    }
    fn get_mut_p<T: ToPrimitive, P: IntoTuple2<T>>(&mut self, p: P) -> &mut Self::Item {
        self.try_get_mut_p(p).expect("[Get2d::get_p] Invalid index")
    }
    fn try_get_mut_xy<T: ToPrimitive>(&mut self, x: T, y: T)
        -> Result<&mut Self::Item, IndexError>;
    fn try_get_mut_p<T: ToPrimitive, P: IntoTuple2<T>>(
        &mut self,
        p: P,
    ) -> Result<&mut Self::Item, IndexError> {
        let t = p.into_tuple2();
        self.try_get_mut_xy(t.0, t.1)
    }
}

impl<D> Get2D for Vec<Vec<D>> {
    type Item = D;
    fn try_get_xy<T: ToPrimitive>(&self, x: T, y: T) -> Result<&Self::Item, IndexError> {
        let x = match x.to_usize() {
            Some(x) => x,
            None => return Err(IndexError::X(x.to_i64().unwrap())),
        };
        let y = match y.to_usize() {
            Some(y) => y,
            None => return Err(IndexError::Y(y.to_i64().unwrap())),
        };
        let line = match self.get(y) {
            Some(l) => l,
            None => return Err(IndexError::Y(y as i64)),
        };
        match line.get(x) {
            Some(res) => Ok(res),
            None => Err(IndexError::X(x as i64)),
        }
    }
}

impl<D> GetMut2D for Vec<Vec<D>> {
    fn try_get_mut_xy<T: ToPrimitive>(
        &mut self,
        x: T,
        y: T,
    ) -> Result<&mut Self::Item, IndexError> {
        let x = match x.to_usize() {
            Some(x) => x,
            None => return Err(IndexError::X(x.to_i64().unwrap())),
        };
        let y = match y.to_usize() {
            Some(y) => y,
            None => return Err(IndexError::Y(y.to_i64().unwrap())),
        };
        let line = match self.get_mut(y) {
            Some(l) => l,
            None => return Err(IndexError::Y(y as i64)),
        };
        match line.get_mut(x) {
            Some(res) => Ok(res),
            None => Err(IndexError::X(x as i64)),
        }
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
            *dest.try_get_mut_p(d)? = convert(source.try_get_p(s)?);
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
            *dest.try_get_mut_p(d)? = source.try_get_p(s)?.clone();
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
            *dest.try_get_mut_p(d)? = convert(source.try_get_p(s)?);
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
            *dest.try_get_mut_p(d)? = source.try_get_p(s)?.clone();
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
    fn try_get_xy<T: ToPrimitive>(&self, x: T, y: T) -> Result<&Self::Item, IndexError> {
        let (x, y) = (x.to_u32().unwrap(), y.to_u32().unwrap());
        if x >= self.width() {
            return Err(IndexError::X(i64::from(x)));
        }
        if y >= self.height() {
            return Err(IndexError::Y(i64::from(y)));
        }
        Ok(self.get_pixel(x, y))
    }
}

#[cfg(feature = "image")]
impl<P, C> GetMut2D for ImageBuffer<P, C>
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
    C: Deref<Target = [P::Subpixel]> + DerefMut,
{
    fn try_get_mut_xy<T: ToPrimitive>(
        &mut self,
        x: T,
        y: T,
    ) -> Result<&mut Self::Item, IndexError> {
        let (x, y) = (x.to_u32().unwrap(), y.to_u32().unwrap());
        if x >= self.width() {
            return Err(IndexError::X(i64::from(x)));
        }
        if y >= self.height() {
            return Err(IndexError::Y(i64::from(y)));
        }
        Ok(self.get_pixel_mut(x, y))
    }
}

#[cfg(feature = "ndarray")]
impl<S: Data> Get2D for ArrayBase<S, Ix2> {
    type Item = S::Elem;
    fn try_get_xy<T: ToPrimitive>(&self, x: T, y: T) -> Result<&Self::Item, IndexError> {
        let (x, y) = (x.to_usize().unwrap(), y.to_usize().unwrap());
        let shape = self.shape();
        if x >= shape[1] {
            return Err(IndexError::X(x as i64));
        }
        if y >= shape[0] {
            return Err(IndexError::Y(y as i64));
        }
        Ok(unsafe { self.uget([y, x]) })
    }
}

#[cfg(feature = "ndarray")]
impl<S: DataMut> GetMut2D for ArrayBase<S, Ix2> {
    fn try_get_mut_xy<T: ToPrimitive>(
        &mut self,
        x: T,
        y: T,
    ) -> Result<&mut Self::Item, IndexError> {
        let (x, y) = (x.to_usize().unwrap(), y.to_usize().unwrap());
        {
            let shape = self.shape();
            if x >= shape[1] {
                return Err(IndexError::X(x as i64));
            }
            if y >= shape[0] {
                return Err(IndexError::Y(y as i64));
            }
        }
        Ok(unsafe { self.uget_mut([y, x]) })
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
        assert_eq!(&3, a.get_xy(3, 3));
        assert_eq!(Err(IndexError::Y(7)), a.try_get_xy(5, 7));
    }
    #[test]
    fn test_copy_rect() {
        let mut a = vec![vec![3; 5]; 7];
        let r1 = RectRange::from_ranges(3..5, 2..6).unwrap();
        let b = vec![vec![80; 100]; 100];
        copy_rect(&b, &mut a, r1.clone(), r1.clone()).unwrap();
        r1.into_iter().for_each(|p| assert_eq!(a.get_p(p), &80));
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
    #[test]
    fn test_contains() {
        let r = RectRange::from_ranges(4..7, 3..7).unwrap();
        assert!(r.contains((6, 6)));
        assert!(!r.contains((6, 7)));
    }
    #[test]
    fn test_is_edge() {
        let r = RectRange::from_ranges(4..7, 3..7).unwrap();
        assert!(r.is_edge((6, 6)));
        assert!(r.is_edge((4, 5)));
        assert!(!r.is_edge((4, 7)));
    }
    #[test]
    fn test_index() {
        let r = RectRange::from_ranges(4..7, 3..7).unwrap();
        for i in 0..r.len() {
            let cd = r.nth(i).unwrap();
            assert_eq!(r.index(cd), Some(i));
        }
        assert_eq!(r.index((6, 7)), None);
    }
    #[test]
    fn test_is_vert_edge() {
        let r = RectRange::from_ranges(4..7, 3..7).unwrap();
        assert!(r.is_vert_edge((4, 5)));
        assert!(!r.is_vert_edge((5, 6)));
    }
    #[test]
    fn test_is_horiz_edge() {
        let r = RectRange::from_ranges(4..7, 3..7).unwrap();
        assert!(!r.is_horiz_edge((4, 5)));
        assert!(r.is_horiz_edge((5, 6)));
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray() {
        use super::ndarray::arr2;
        let a = arr2(&[[1, 2], [3, 4]]);
        assert_eq!(a.get_xy(1, 0), &2);
        assert!(a.try_get_xy(4, 0).is_err());
    }
}
