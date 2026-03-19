use crate::allocator::WgpuAllocator;
use kornia_image::{Image, ImageSize};

/// A GPU-resident image.
///
/// Wraps `Image<T, C, WgpuAllocator>` and exposes no CPU slice methods.
/// The only way to get pixel data out is `transfer::image_to_cpu`.
pub struct GpuImage<T, const C: usize>(pub(crate) Image<T, C, WgpuAllocator>);

impl<T, const C: usize> GpuImage<T, C> {
    /// Returns the logical size (dimensions) of the image.
    ///
    /// # Returns
    ///
    /// An [`ImageSize`] representing the width and height.
    pub fn size(&self) -> ImageSize {
        self.0.size()
    }
    /// Returns the number of channels in the image.
    ///
    /// # Returns
    ///
    /// The number of channels, `C`.
    pub fn channels(&self) -> usize {
        C
    }
}

/// A GPU-resident tensor.
///
/// Wraps `Tensor<T, N, WgpuAllocator>` and exposes no CPU slice methods.
/// The only way to get element data out is `session.download_tensor`.
pub struct GpuTensor<T, const N: usize>(pub(crate) kornia_tensor::Tensor<T, N, WgpuAllocator>);

impl<T, const N: usize> GpuTensor<T, N> {
    /// Returns the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A reference to the array of size `N` containing the dimensions.
    pub fn shape(&self) -> &[usize; N] {
        &self.0.shape
    }
    /// Returns the strides of the tensor.
    ///
    /// # Returns
    ///
    /// A reference to the array of size `N` containing the strides.
    pub fn strides(&self) -> &[usize; N] {
        &self.0.strides
    }
}
