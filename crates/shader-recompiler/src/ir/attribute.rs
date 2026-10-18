#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Attribute(u16);

#[allow(non_upper_case_globals)]
impl Attribute {
    pub const PrimitiveId: Self = Self(24);
    pub const Layer: Self = Self(25);
    pub const ViewportIndex: Self = Self(26);
    pub const PointSize: Self = Self(27);
    pub const PositionX: Self = Self(28);
    pub const PositionY: Self = Self(29);
    pub const PositionZ: Self = Self(30);
    pub const PositionW: Self = Self(31);
    pub const Generic0X: Self = Self(32);
    pub const Generic0Y: Self = Self(33);
    pub const Generic0Z: Self = Self(34);
    pub const Generic0W: Self = Self(35);
    pub const Generic1X: Self = Self(36);
    pub const Generic1Y: Self = Self(37);
    pub const Generic1Z: Self = Self(38);
    pub const Generic1W: Self = Self(39);
    pub const Generic2X: Self = Self(40);
    pub const Generic2Y: Self = Self(41);
    pub const Generic2Z: Self = Self(42);
    pub const Generic2W: Self = Self(43);
    pub const Generic3X: Self = Self(44);
    pub const Generic3Y: Self = Self(45);
    pub const Generic3Z: Self = Self(46);
    pub const Generic3W: Self = Self(47);
    pub const Generic4X: Self = Self(48);
    pub const Generic4Y: Self = Self(49);
    pub const Generic4Z: Self = Self(50);
    pub const Generic4W: Self = Self(51);
    pub const Generic5X: Self = Self(52);
    pub const Generic5Y: Self = Self(53);
    pub const Generic5Z: Self = Self(54);
    pub const Generic5W: Self = Self(55);
    pub const Generic6X: Self = Self(56);
    pub const Generic6Y: Self = Self(57);
    pub const Generic6Z: Self = Self(58);
    pub const Generic6W: Self = Self(59);
    pub const Generic7X: Self = Self(60);
    pub const Generic7Y: Self = Self(61);
    pub const Generic7Z: Self = Self(62);
    pub const Generic7W: Self = Self(63);
    pub const Generic8X: Self = Self(64);
    pub const Generic8Y: Self = Self(65);
    pub const Generic8Z: Self = Self(66);
    pub const Generic8W: Self = Self(67);
    pub const Generic9X: Self = Self(68);
    pub const Generic9Y: Self = Self(69);
    pub const Generic9Z: Self = Self(70);
    pub const Generic9W: Self = Self(71);
    pub const Generic10X: Self = Self(72);
    pub const Generic10Y: Self = Self(73);
    pub const Generic10Z: Self = Self(74);
    pub const Generic10W: Self = Self(75);
    pub const Generic11X: Self = Self(76);
    pub const Generic11Y: Self = Self(77);
    pub const Generic11Z: Self = Self(78);
    pub const Generic11W: Self = Self(79);
    pub const Generic12X: Self = Self(80);
    pub const Generic12Y: Self = Self(81);
    pub const Generic12Z: Self = Self(82);
    pub const Generic12W: Self = Self(83);
    pub const Generic13X: Self = Self(84);
    pub const Generic13Y: Self = Self(85);
    pub const Generic13Z: Self = Self(86);
    pub const Generic13W: Self = Self(87);
    pub const Generic14X: Self = Self(88);
    pub const Generic14Y: Self = Self(89);
    pub const Generic14Z: Self = Self(90);
    pub const Generic14W: Self = Self(91);
    pub const Generic15X: Self = Self(92);
    pub const Generic15Y: Self = Self(93);
    pub const Generic15Z: Self = Self(94);
    pub const Generic15W: Self = Self(95);
    pub const Generic16X: Self = Self(96);
    pub const Generic16Y: Self = Self(97);
    pub const Generic16Z: Self = Self(98);
    pub const Generic16W: Self = Self(99);
    pub const Generic17X: Self = Self(100);
    pub const Generic17Y: Self = Self(101);
    pub const Generic17Z: Self = Self(102);
    pub const Generic17W: Self = Self(103);
    pub const Generic18X: Self = Self(104);
    pub const Generic18Y: Self = Self(105);
    pub const Generic18Z: Self = Self(106);
    pub const Generic18W: Self = Self(107);
    pub const Generic19X: Self = Self(108);
    pub const Generic19Y: Self = Self(109);
    pub const Generic19Z: Self = Self(110);
    pub const Generic19W: Self = Self(111);
    pub const Generic20X: Self = Self(112);
    pub const Generic20Y: Self = Self(113);
    pub const Generic20Z: Self = Self(114);
    pub const Generic20W: Self = Self(115);
    pub const Generic21X: Self = Self(116);
    pub const Generic21Y: Self = Self(117);
    pub const Generic21Z: Self = Self(118);
    pub const Generic21W: Self = Self(119);
    pub const Generic22X: Self = Self(120);
    pub const Generic22Y: Self = Self(121);
    pub const Generic22Z: Self = Self(122);
    pub const Generic22W: Self = Self(123);
    pub const Generic23X: Self = Self(124);
    pub const Generic23Y: Self = Self(125);
    pub const Generic23Z: Self = Self(126);
    pub const Generic23W: Self = Self(127);
    pub const Generic24X: Self = Self(128);
    pub const Generic24Y: Self = Self(129);
    pub const Generic24Z: Self = Self(130);
    pub const Generic24W: Self = Self(131);
    pub const Generic25X: Self = Self(132);
    pub const Generic25Y: Self = Self(133);
    pub const Generic25Z: Self = Self(134);
    pub const Generic25W: Self = Self(135);
    pub const Generic26X: Self = Self(136);
    pub const Generic26Y: Self = Self(137);
    pub const Generic26Z: Self = Self(138);
    pub const Generic26W: Self = Self(139);
    pub const Generic27X: Self = Self(140);
    pub const Generic27Y: Self = Self(141);
    pub const Generic27Z: Self = Self(142);
    pub const Generic27W: Self = Self(143);
    pub const Generic28X: Self = Self(144);
    pub const Generic28Y: Self = Self(145);
    pub const Generic28Z: Self = Self(146);
    pub const Generic28W: Self = Self(147);
    pub const Generic29X: Self = Self(148);
    pub const Generic29Y: Self = Self(149);
    pub const Generic29Z: Self = Self(150);
    pub const Generic29W: Self = Self(151);
    pub const Generic30X: Self = Self(152);
    pub const Generic30Y: Self = Self(153);
    pub const Generic30Z: Self = Self(154);
    pub const Generic30W: Self = Self(155);
    pub const Generic31X: Self = Self(156);
    pub const Generic31Y: Self = Self(157);
    pub const Generic31Z: Self = Self(158);
    pub const Generic31W: Self = Self(159);
    pub const ColorFrontDiffuseR: Self = Self(160);
    pub const ColorFrontDiffuseG: Self = Self(161);
    pub const ColorFrontDiffuseB: Self = Self(162);
    pub const ColorFrontDiffuseA: Self = Self(163);
    pub const ColorFrontSpecularR: Self = Self(164);
    pub const ColorFrontSpecularG: Self = Self(165);
    pub const ColorFrontSpecularB: Self = Self(166);
    pub const ColorFrontSpecularA: Self = Self(167);
    pub const ColorBackDiffuseR: Self = Self(168);
    pub const ColorBackDiffuseG: Self = Self(169);
    pub const ColorBackDiffuseB: Self = Self(170);
    pub const ColorBackDiffuseA: Self = Self(171);
    pub const ColorBackSpecularR: Self = Self(172);
    pub const ColorBackSpecularG: Self = Self(173);
    pub const ColorBackSpecularB: Self = Self(174);
    pub const ColorBackSpecularA: Self = Self(175);
    pub const ClipDistance0: Self = Self(176);
    pub const ClipDistance1: Self = Self(177);
    pub const ClipDistance2: Self = Self(178);
    pub const ClipDistance3: Self = Self(179);
    pub const ClipDistance4: Self = Self(180);
    pub const ClipDistance5: Self = Self(181);
    pub const ClipDistance6: Self = Self(182);
    pub const ClipDistance7: Self = Self(183);
    pub const PointSpriteS: Self = Self(184);
    pub const PointSpriteT: Self = Self(185);
    pub const FogCoordinate: Self = Self(186);
    pub const TessellationEvaluationPointU: Self = Self(188);
    pub const TessellationEvaluationPointV: Self = Self(189);
    pub const InstanceId: Self = Self(190);
    pub const VertexId: Self = Self(191);
    pub const FixedFncTexture0S: Self = Self(192);
    pub const FixedFncTexture0T: Self = Self(193);
    pub const FixedFncTexture0R: Self = Self(194);
    pub const FixedFncTexture0Q: Self = Self(195);
    pub const FixedFncTexture1S: Self = Self(196);
    pub const FixedFncTexture1T: Self = Self(197);
    pub const FixedFncTexture1R: Self = Self(198);
    pub const FixedFncTexture1Q: Self = Self(199);
    pub const FixedFncTexture2S: Self = Self(200);
    pub const FixedFncTexture2T: Self = Self(201);
    pub const FixedFncTexture2R: Self = Self(202);
    pub const FixedFncTexture2Q: Self = Self(203);
    pub const FixedFncTexture3S: Self = Self(204);
    pub const FixedFncTexture3T: Self = Self(205);
    pub const FixedFncTexture3R: Self = Self(206);
    pub const FixedFncTexture3Q: Self = Self(207);
    pub const FixedFncTexture4S: Self = Self(208);
    pub const FixedFncTexture4T: Self = Self(209);
    pub const FixedFncTexture4R: Self = Self(210);
    pub const FixedFncTexture4Q: Self = Self(211);
    pub const FixedFncTexture5S: Self = Self(212);
    pub const FixedFncTexture5T: Self = Self(213);
    pub const FixedFncTexture5R: Self = Self(214);
    pub const FixedFncTexture5Q: Self = Self(215);
    pub const FixedFncTexture6S: Self = Self(216);
    pub const FixedFncTexture6T: Self = Self(217);
    pub const FixedFncTexture6R: Self = Self(218);
    pub const FixedFncTexture6Q: Self = Self(219);
    pub const FixedFncTexture7S: Self = Self(220);
    pub const FixedFncTexture7T: Self = Self(221);
    pub const FixedFncTexture7R: Self = Self(222);
    pub const FixedFncTexture7Q: Self = Self(223);
    pub const FixedFncTexture8S: Self = Self(224);
    pub const FixedFncTexture8T: Self = Self(225);
    pub const FixedFncTexture8R: Self = Self(226);
    pub const FixedFncTexture8Q: Self = Self(227);
    pub const FixedFncTexture9S: Self = Self(228);
    pub const FixedFncTexture9T: Self = Self(229);
    pub const FixedFncTexture9R: Self = Self(230);
    pub const FixedFncTexture9Q: Self = Self(231);
    pub const ViewportMask: Self = Self(232);
    pub const FrontFace: Self = Self(255);
    pub const BaseInstance: Self = Self(256);
    pub const BaseVertex: Self = Self(257);
    pub const DrawID: Self = Self(258);

    /// All 512 varying slots are representable, including reserved attributes.
    pub const fn from_index(index: u16) -> Option<Self> {
        if index < 512 { Some(Self(index)) } else { None }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn generic(index: usize, component: usize) -> Option<Self> {
        if index < 32 && component < 4 {
            Some(Self(32 + (index * 4 + component) as u16))
        } else {
            None
        }
    }

    /// Preserves the reference's inclusive Generic31X bound, including its
    /// exclusion of Generic31Y/Z/W. This is not the varying-mask range test.
    pub const fn is_generic(self) -> bool {
        self.0 >= 32 && self.0 <= 156
    }

    pub const fn is_legacy(self) -> bool {
        (self.0 >= 160 && self.0 <= 175) || self.0 == 186 || (self.0 >= 192 && self.0 <= 231)
    }

    pub fn generic_index(self) -> crate::Result<u32> {
        if self.is_generic() {
            Ok((u32::from(self.0) - 32) / 4)
        } else {
            Err(crate::CompileError::new(
                crate::diagnostics::FailureCategory::InvalidArgument,
                format!("Attribute {} is not generic", self.0),
            ))
        }
    }

    pub fn generic_element(self) -> crate::Result<u32> {
        self.generic_index().map(|_| u32::from(self.0) % 4)
    }
}
