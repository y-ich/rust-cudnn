//! Defines a Filter Descriptor.
//!
//! A Filter Descriptor is used to hold information about the Filter,
//! which is needed for forward and backward convolutional operations.

use super::{API, Error};
use super::utils::{DataType, TensorFormat};
use ffi::*;

#[derive(Debug, Clone)]
/// Describes a Filter Descriptor.
pub struct FilterDescriptor {
    id: cudnnFilterDescriptor_t,
}

impl Drop for FilterDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_filter_descriptor(*self.id_c());
    }
}

impl FilterDescriptor {
    /// Initializes a new CUDA cuDNN FilterDescriptor.
    pub fn new(filter_dim: &[i32], data_type: DataType, format: TensorFormat) -> Result<FilterDescriptor, Error> {
        let nb_dims = filter_dim.len() as i32;

        let generic_filter_desc = try!(API::create_filter_descriptor());
        let d_type;
        match data_type {
            DataType::Float => {
                d_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
            },
            DataType::Double => {
                d_type = cudnnDataType_t::CUDNN_DATA_DOUBLE;
            },
            DataType::Half => {
                d_type = cudnnDataType_t::CUDNN_DATA_HALF;
            }
        }
        let t_format;
        match format {
            TensorFormat::NCHW => {
                t_format = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
            },
            TensorFormat::NHWC => {
                t_format = cudnnTensorFormat_t::CUDNN_TENSOR_NHWC;
            },
            TensorFormat::NCHW_VECT_C => {
                t_format = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW_VECT_C;
            }
        }
        try!(API::set_filter_descriptor(generic_filter_desc, d_type, t_format, nb_dims, filter_dim.as_ptr()));
        Ok(FilterDescriptor::from_c(generic_filter_desc))
    }

    /// Initializes a new CUDA cuDNN FilterDescriptor from its C type.
    pub fn from_c(id: cudnnFilterDescriptor_t) -> FilterDescriptor {
        FilterDescriptor { id: id }
    }

    /// Returns the CUDA cuDNN FilterDescriptor as its C type.
    pub fn id_c(&self) -> &cudnnFilterDescriptor_t {
        &self.id
    }
}
