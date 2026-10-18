/// Guest virtual byte address; never an instruction index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct GuestAddress(pub u64);

/// Environment-relative byte location, including any graphics header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ProgramOffset(pub u32);

/// Byte offset within a constant buffer or other memory object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ByteOffset(pub u32);

/// Guest constant-buffer bank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CbufIndex(pub u32);

/// Index into a typed resource descriptor collection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResourceIndex(pub u32);

/// Emitted host descriptor set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DescriptorSet(pub u32);

/// Emitted host binding, distinct from a guest resource index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DescriptorBinding(pub u32);

/// Raw guest texture handle consumed during compilation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TextureHandle(pub u32);

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum Stage {
    #[default]
    VertexB = 0,
    TessellationControl = 1,
    TessellationEval = 2,
    Geometry = 3,
    Fragment = 4,
    Compute = 5,
    VertexA = 6,
}

impl TryFrom<u32> for Stage {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::VertexB),
            1 => Ok(Self::TessellationControl),
            2 => Ok(Self::TessellationEval),
            3 => Ok(Self::Geometry),
            4 => Ok(Self::Fragment),
            5 => Ok(Self::Compute),
            6 => Ok(Self::VertexA),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid Stage value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum OutputTopology {
    PointList = 1,
    LineStrip = 6,
    TriangleStrip = 7,
}

impl TryFrom<u32> for OutputTopology {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            1 => Ok(Self::PointList),
            6 => Ok(Self::LineStrip),
            7 => Ok(Self::TriangleStrip),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid OutputTopology value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PixelImap {
    #[default]
    Unused = 0,
    Constant = 1,
    Perspective = 2,
    ScreenLinear = 3,
}

impl TryFrom<u32> for PixelImap {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::Unused),
            1 => Ok(Self::Constant),
            2 => Ok(Self::Perspective),
            3 => Ok(Self::ScreenLinear),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid PixelImap value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum AttributeType {
    #[default]
    Float = 0,
    SignedInt = 1,
    UnsignedInt = 2,
    SignedScaled = 3,
    UnsignedScaled = 4,
    Disabled = 5,
}

impl TryFrom<u32> for AttributeType {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::Float),
            1 => Ok(Self::SignedInt),
            2 => Ok(Self::UnsignedInt),
            3 => Ok(Self::SignedScaled),
            4 => Ok(Self::UnsignedScaled),
            5 => Ok(Self::Disabled),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid AttributeType value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum InputTopology {
    #[default]
    Points = 0,
    Lines = 1,
    LinesAdjacency = 2,
    Triangles = 3,
    TrianglesAdjacency = 4,
}

impl TryFrom<u32> for InputTopology {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::Points),
            1 => Ok(Self::Lines),
            2 => Ok(Self::LinesAdjacency),
            3 => Ok(Self::Triangles),
            4 => Ok(Self::TrianglesAdjacency),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid InputTopology value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum CompareFunction {
    #[default]
    Never = 0,
    Less = 1,
    Equal = 2,
    LessThanEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterThanEqual = 6,
    Always = 7,
}

impl TryFrom<u32> for CompareFunction {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::Never),
            1 => Ok(Self::Less),
            2 => Ok(Self::Equal),
            3 => Ok(Self::LessThanEqual),
            4 => Ok(Self::Greater),
            5 => Ok(Self::NotEqual),
            6 => Ok(Self::GreaterThanEqual),
            7 => Ok(Self::Always),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid CompareFunction value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum TessPrimitive {
    #[default]
    Isolines = 0,
    Triangles = 1,
    Quads = 2,
}

impl TryFrom<u32> for TessPrimitive {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::Isolines),
            1 => Ok(Self::Triangles),
            2 => Ok(Self::Quads),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid TessPrimitive value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum TessSpacing {
    #[default]
    Equal = 0,
    FractionalOdd = 1,
    FractionalEven = 2,
}

impl TryFrom<u32> for TessSpacing {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::Equal),
            1 => Ok(Self::FractionalOdd),
            2 => Ok(Self::FractionalEven),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid TessSpacing value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum ReplaceConstant {
    #[default]
    BaseInstance = 0,
    BaseVertex = 1,
    DrawID = 2,
}

impl TryFrom<u32> for ReplaceConstant {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::BaseInstance),
            1 => Ok(Self::BaseVertex),
            2 => Ok(Self::DrawID),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid ReplaceConstant value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum TextureType {
    #[default]
    Color1D = 0,
    ColorArray1D = 1,
    Color2D = 2,
    ColorArray2D = 3,
    Color3D = 4,
    ColorCube = 5,
    ColorArrayCube = 6,
    Buffer = 7,
    Color2DRect = 8,
}

impl TryFrom<u32> for TextureType {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::Color1D),
            1 => Ok(Self::ColorArray1D),
            2 => Ok(Self::Color2D),
            3 => Ok(Self::ColorArray2D),
            4 => Ok(Self::Color3D),
            5 => Ok(Self::ColorCube),
            6 => Ok(Self::ColorArrayCube),
            7 => Ok(Self::Buffer),
            8 => Ok(Self::Color2DRect),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid TextureType value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum TexturePixelFormat {
    #[default]
    A8B8G8R8_UNORM = 0,
    A8B8G8R8_SNORM = 1,
    A8B8G8R8_SINT = 2,
    A8B8G8R8_UINT = 3,
    R5G6B5_UNORM = 4,
    B5G6R5_UNORM = 5,
    A1R5G5B5_UNORM = 6,
    A2B10G10R10_UNORM = 7,
    A2B10G10R10_UINT = 8,
    A2R10G10B10_UNORM = 9,
    A1B5G5R5_UNORM = 10,
    A5B5G5R1_UNORM = 11,
    R8_UNORM = 12,
    R8_SNORM = 13,
    R8_SINT = 14,
    R8_UINT = 15,
    R16G16B16A16_FLOAT = 16,
    R16G16B16A16_UNORM = 17,
    R16G16B16A16_SNORM = 18,
    R16G16B16A16_SINT = 19,
    R16G16B16A16_UINT = 20,
    B10G11R11_FLOAT = 21,
    R32G32B32A32_UINT = 22,
    BC1_RGBA_UNORM = 23,
    BC2_UNORM = 24,
    BC3_UNORM = 25,
    BC4_UNORM = 26,
    BC4_SNORM = 27,
    BC5_UNORM = 28,
    BC5_SNORM = 29,
    BC7_UNORM = 30,
    BC6H_UFLOAT = 31,
    BC6H_SFLOAT = 32,
    ASTC_2D_4X4_UNORM = 33,
    B8G8R8A8_UNORM = 34,
    R32G32B32A32_FLOAT = 35,
    R32G32B32A32_SINT = 36,
    R32G32_FLOAT = 37,
    R32G32_SINT = 38,
    R32_FLOAT = 39,
    R16_FLOAT = 40,
    R16_UNORM = 41,
    R16_SNORM = 42,
    R16_UINT = 43,
    R16_SINT = 44,
    R16G16_UNORM = 45,
    R16G16_FLOAT = 46,
    R16G16_UINT = 47,
    R16G16_SINT = 48,
    R16G16_SNORM = 49,
    R32G32B32_FLOAT = 50,
    A8B8G8R8_SRGB = 51,
    R8G8_UNORM = 52,
    R8G8_SNORM = 53,
    R8G8_SINT = 54,
    R8G8_UINT = 55,
    R32G32_UINT = 56,
    R16G16B16X16_FLOAT = 57,
    R32_UINT = 58,
    R32_SINT = 59,
    ASTC_2D_8X8_UNORM = 60,
    ASTC_2D_8X5_UNORM = 61,
    ASTC_2D_5X4_UNORM = 62,
    B8G8R8A8_SRGB = 63,
    BC1_RGBA_SRGB = 64,
    BC2_SRGB = 65,
    BC3_SRGB = 66,
    BC7_SRGB = 67,
    A4B4G4R4_UNORM = 68,
    G4R4_UNORM = 69,
    ASTC_2D_4X4_SRGB = 70,
    ASTC_2D_8X8_SRGB = 71,
    ASTC_2D_8X5_SRGB = 72,
    ASTC_2D_5X4_SRGB = 73,
    ASTC_2D_5X5_UNORM = 74,
    ASTC_2D_5X5_SRGB = 75,
    ASTC_2D_10X8_UNORM = 76,
    ASTC_2D_10X8_SRGB = 77,
    ASTC_2D_6X6_UNORM = 78,
    ASTC_2D_6X6_SRGB = 79,
    ASTC_2D_10X6_UNORM = 80,
    ASTC_2D_10X6_SRGB = 81,
    ASTC_2D_10X5_UNORM = 82,
    ASTC_2D_10X5_SRGB = 83,
    ASTC_2D_10X10_UNORM = 84,
    ASTC_2D_10X10_SRGB = 85,
    ASTC_2D_12X10_UNORM = 86,
    ASTC_2D_12X10_SRGB = 87,
    ASTC_2D_12X12_UNORM = 88,
    ASTC_2D_12X12_SRGB = 89,
    ASTC_2D_8X6_UNORM = 90,
    ASTC_2D_8X6_SRGB = 91,
    ASTC_2D_6X5_UNORM = 92,
    ASTC_2D_6X5_SRGB = 93,
    E5B9G9R9_FLOAT = 94,
    D32_FLOAT = 95,
    D16_UNORM = 96,
    X8_D24_UNORM = 97,
    S8_UINT = 98,
    D24_UNORM_S8_UINT = 99,
    S8_UINT_D24_UNORM = 100,
    D32_FLOAT_S8_UINT = 101,
}

impl TryFrom<u32> for TexturePixelFormat {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::A8B8G8R8_UNORM),
            1 => Ok(Self::A8B8G8R8_SNORM),
            2 => Ok(Self::A8B8G8R8_SINT),
            3 => Ok(Self::A8B8G8R8_UINT),
            4 => Ok(Self::R5G6B5_UNORM),
            5 => Ok(Self::B5G6R5_UNORM),
            6 => Ok(Self::A1R5G5B5_UNORM),
            7 => Ok(Self::A2B10G10R10_UNORM),
            8 => Ok(Self::A2B10G10R10_UINT),
            9 => Ok(Self::A2R10G10B10_UNORM),
            10 => Ok(Self::A1B5G5R5_UNORM),
            11 => Ok(Self::A5B5G5R1_UNORM),
            12 => Ok(Self::R8_UNORM),
            13 => Ok(Self::R8_SNORM),
            14 => Ok(Self::R8_SINT),
            15 => Ok(Self::R8_UINT),
            16 => Ok(Self::R16G16B16A16_FLOAT),
            17 => Ok(Self::R16G16B16A16_UNORM),
            18 => Ok(Self::R16G16B16A16_SNORM),
            19 => Ok(Self::R16G16B16A16_SINT),
            20 => Ok(Self::R16G16B16A16_UINT),
            21 => Ok(Self::B10G11R11_FLOAT),
            22 => Ok(Self::R32G32B32A32_UINT),
            23 => Ok(Self::BC1_RGBA_UNORM),
            24 => Ok(Self::BC2_UNORM),
            25 => Ok(Self::BC3_UNORM),
            26 => Ok(Self::BC4_UNORM),
            27 => Ok(Self::BC4_SNORM),
            28 => Ok(Self::BC5_UNORM),
            29 => Ok(Self::BC5_SNORM),
            30 => Ok(Self::BC7_UNORM),
            31 => Ok(Self::BC6H_UFLOAT),
            32 => Ok(Self::BC6H_SFLOAT),
            33 => Ok(Self::ASTC_2D_4X4_UNORM),
            34 => Ok(Self::B8G8R8A8_UNORM),
            35 => Ok(Self::R32G32B32A32_FLOAT),
            36 => Ok(Self::R32G32B32A32_SINT),
            37 => Ok(Self::R32G32_FLOAT),
            38 => Ok(Self::R32G32_SINT),
            39 => Ok(Self::R32_FLOAT),
            40 => Ok(Self::R16_FLOAT),
            41 => Ok(Self::R16_UNORM),
            42 => Ok(Self::R16_SNORM),
            43 => Ok(Self::R16_UINT),
            44 => Ok(Self::R16_SINT),
            45 => Ok(Self::R16G16_UNORM),
            46 => Ok(Self::R16G16_FLOAT),
            47 => Ok(Self::R16G16_UINT),
            48 => Ok(Self::R16G16_SINT),
            49 => Ok(Self::R16G16_SNORM),
            50 => Ok(Self::R32G32B32_FLOAT),
            51 => Ok(Self::A8B8G8R8_SRGB),
            52 => Ok(Self::R8G8_UNORM),
            53 => Ok(Self::R8G8_SNORM),
            54 => Ok(Self::R8G8_SINT),
            55 => Ok(Self::R8G8_UINT),
            56 => Ok(Self::R32G32_UINT),
            57 => Ok(Self::R16G16B16X16_FLOAT),
            58 => Ok(Self::R32_UINT),
            59 => Ok(Self::R32_SINT),
            60 => Ok(Self::ASTC_2D_8X8_UNORM),
            61 => Ok(Self::ASTC_2D_8X5_UNORM),
            62 => Ok(Self::ASTC_2D_5X4_UNORM),
            63 => Ok(Self::B8G8R8A8_SRGB),
            64 => Ok(Self::BC1_RGBA_SRGB),
            65 => Ok(Self::BC2_SRGB),
            66 => Ok(Self::BC3_SRGB),
            67 => Ok(Self::BC7_SRGB),
            68 => Ok(Self::A4B4G4R4_UNORM),
            69 => Ok(Self::G4R4_UNORM),
            70 => Ok(Self::ASTC_2D_4X4_SRGB),
            71 => Ok(Self::ASTC_2D_8X8_SRGB),
            72 => Ok(Self::ASTC_2D_8X5_SRGB),
            73 => Ok(Self::ASTC_2D_5X4_SRGB),
            74 => Ok(Self::ASTC_2D_5X5_UNORM),
            75 => Ok(Self::ASTC_2D_5X5_SRGB),
            76 => Ok(Self::ASTC_2D_10X8_UNORM),
            77 => Ok(Self::ASTC_2D_10X8_SRGB),
            78 => Ok(Self::ASTC_2D_6X6_UNORM),
            79 => Ok(Self::ASTC_2D_6X6_SRGB),
            80 => Ok(Self::ASTC_2D_10X6_UNORM),
            81 => Ok(Self::ASTC_2D_10X6_SRGB),
            82 => Ok(Self::ASTC_2D_10X5_UNORM),
            83 => Ok(Self::ASTC_2D_10X5_SRGB),
            84 => Ok(Self::ASTC_2D_10X10_UNORM),
            85 => Ok(Self::ASTC_2D_10X10_SRGB),
            86 => Ok(Self::ASTC_2D_12X10_UNORM),
            87 => Ok(Self::ASTC_2D_12X10_SRGB),
            88 => Ok(Self::ASTC_2D_12X12_UNORM),
            89 => Ok(Self::ASTC_2D_12X12_SRGB),
            90 => Ok(Self::ASTC_2D_8X6_UNORM),
            91 => Ok(Self::ASTC_2D_8X6_SRGB),
            92 => Ok(Self::ASTC_2D_6X5_UNORM),
            93 => Ok(Self::ASTC_2D_6X5_SRGB),
            94 => Ok(Self::E5B9G9R9_FLOAT),
            95 => Ok(Self::D32_FLOAT),
            96 => Ok(Self::D16_UNORM),
            97 => Ok(Self::X8_D24_UNORM),
            98 => Ok(Self::S8_UINT),
            99 => Ok(Self::D24_UNORM_S8_UINT),
            100 => Ok(Self::S8_UINT_D24_UNORM),
            101 => Ok(Self::D32_FLOAT_S8_UINT),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid TexturePixelFormat value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum ImageFormat {
    #[default]
    Typeless = 0,
    R8_UINT = 1,
    R8_SINT = 2,
    R16_UINT = 3,
    R16_SINT = 4,
    R32_UINT = 5,
    R32G32_UINT = 6,
    R32G32B32A32_UINT = 7,
}

impl TryFrom<u32> for ImageFormat {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::Typeless),
            1 => Ok(Self::R8_UINT),
            2 => Ok(Self::R8_SINT),
            3 => Ok(Self::R16_UINT),
            4 => Ok(Self::R16_SINT),
            5 => Ok(Self::R32_UINT),
            6 => Ok(Self::R32G32_UINT),
            7 => Ok(Self::R32G32B32A32_UINT),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid ImageFormat value {value:#x}"),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum Interpolation {
    #[default]
    Smooth = 0,
    Flat = 1,
    NoPerspective = 2,
}

impl TryFrom<u32> for Interpolation {
    type Error = crate::CompileError;
    fn try_from(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::Smooth),
            1 => Ok(Self::Flat),
            2 => Ok(Self::NoPerspective),
            _ => Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Invalid Interpolation value {value:#x}"),
            )),
        }
    }
}

/// Host stage slots; Vertex A shares the vertex pipeline slot with Vertex B.
pub const MAX_STAGE_TYPES: usize = 6;
pub const NUM_TEXTURE_TYPES: usize = 9;

impl InputTopology {
    pub const fn vertices(self) -> u32 {
        match self {
            Self::Points => 1,
            Self::Lines => 2,
            Self::LinesAdjacency => 4,
            Self::Triangles => 3,
            Self::TrianglesAdjacency => 6,
        }
    }
}
