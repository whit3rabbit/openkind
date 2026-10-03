use std::{fmt, str::FromStr};

/// A physical PCI domain, bus, device, and function address.
///
/// Parse the canonical `0000:03:00.0` form. This address is independent of
/// runtime device ordering and GPU visibility masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PciAddress {
    domain: u16,
    bus: u8,
    device: u8,
    function: u8,
}

/// A PCI address was malformed or contained an out-of-range device/function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("expected a PCI address in the form 0000:03:00.0")]
pub struct ParsePciAddressError;

impl FromStr for PciAddress {
    type Err = ParsePciAddressError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let bytes = value.as_bytes();
        if bytes.len() != 12
            || !value.is_ascii()
            || bytes[4] != b':'
            || bytes[7] != b':'
            || bytes[10] != b'.'
            || bytes
                .iter()
                .enumerate()
                .any(|(index, byte)| !matches!(index, 4 | 7 | 10) && !byte.is_ascii_hexdigit())
        {
            return Err(ParsePciAddressError);
        }
        let address = Self {
            domain: u16::from_str_radix(&value[..4], 16).map_err(|_| ParsePciAddressError)?,
            bus: u8::from_str_radix(&value[5..7], 16).map_err(|_| ParsePciAddressError)?,
            device: u8::from_str_radix(&value[8..10], 16).map_err(|_| ParsePciAddressError)?,
            function: u8::from_str_radix(&value[11..], 16).map_err(|_| ParsePciAddressError)?,
        };
        if address.device > 31 || address.function > 7 {
            return Err(ParsePciAddressError);
        }
        Ok(address)
    }
}

impl fmt::Display for PciAddress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

#[cfg(any(target_os = "linux", all(test, unix)))]
use std::{collections::BTreeMap, fs, path::Path};

#[cfg(any(target_os = "linux", all(test, unix)))]
use crate::{Accelerator, DeviceType};

/// KFD membership proves kernel admission; PCI vendor/class alone does not.
#[cfg(any(target_os = "linux", all(test, unix)))]
pub(crate) fn probe(sys_root: &Path, kfd_device: &Path) -> Vec<Accelerator> {
    if !kfd_device.exists() {
        return Vec::new();
    }
    let Ok(nodes) = fs::read_dir(sys_root.join("class/kfd/kfd/topology/nodes")) else {
        return Vec::new();
    };
    let mut nodes: Vec<_> = nodes.flatten().map(|entry| entry.path()).collect();
    nodes.sort();
    let mut gpus: BTreeMap<PciAddress, Option<String>> = BTreeMap::new();
    for node in nodes {
        let gpu_id = read_number(&node.join("gpu_id"));
        if !gpu_id.is_some_and(|id| id > 0) {
            continue;
        }
        let Ok(properties) = fs::read_to_string(node.join("properties")) else {
            continue;
        };
        if property(&properties, "vendor_id") != Some(0x1002)
            || !property(&properties, "simd_count").is_some_and(|count| count > 0)
        {
            continue;
        }
        let Some(address) = node_pci_address(sys_root, &properties) else {
            continue;
        };
        let pci = sys_root.join("bus/pci/devices").join(address.to_string());
        let vendor = read_hex(&pci.join("vendor"));
        let class = read_hex(&pci.join("class")).map(|class| class >> 16);
        // Instinct compute accelerators use class 0x12 rather than 0x03.
        if vendor != Some(0x1002) || !matches!(class, Some(0x03 | 0x12)) {
            continue;
        }
        let name = fs::read_to_string(node.join("name"))
            .ok()
            .map(|name| name.trim().to_owned())
            .filter(|name| !name.is_empty());
        gpus.entry(address)
            .and_modify(|existing_name| {
                if existing_name.is_none() {
                    existing_name.clone_from(&name);
                }
            })
            .or_insert(name);
    }
    gpus.into_iter()
        .map(|(pci_address, name)| {
            let pci = sys_root
                .join("bus/pci/devices")
                .join(pci_address.to_string());
            // KFD bank sizes can describe a partition, even if other nodes
            // are inaccessible. Only the physical PCI attribute is per-card.
            let total_memory_bytes = read_number(&pci.join("mem_info_vram_total"));
            Accelerator {
                device: DeviceType::Rocm { pci_address },
                name: name.unwrap_or_else(|| format!("AMD GPU ({pci_address})")),
                total_memory_bytes,
            }
        })
        .collect()
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn node_pci_address(sys_root: &Path, properties: &str) -> Option<PciAddress> {
    // KFD partitions encode node_id into location_id's low bits. The DRM
    // render device links back to their actual physical PCI function.
    if let Some(minor) = property(properties, "drm_render_minor") {
        let device = sys_root.join(format!("class/drm/renderD{minor}/device"));
        // An unreadable DRM mapping cannot prove which physical function
        // owns a partition. Its location_id may name a different device.
        let target = fs::canonicalize(device).ok()?;
        return target.file_name()?.to_str()?.parse().ok();
    }
    let domain = u16::try_from(property(properties, "domain")?).ok()?;
    let location = u16::try_from(property(properties, "location_id")?).ok()?;
    Some(PciAddress {
        domain,
        bus: (location >> 8) as u8,
        device: ((location >> 3) & 31) as u8,
        function: (location & 7) as u8,
    })
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn property(properties: &str, key: &str) -> Option<u64> {
    properties.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        if fields.next()? != key {
            return None;
        }
        fields.next()?.parse().ok()
    })
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn read_number(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn read_hex(path: &Path) -> Option<u64> {
    u64::from_str_radix(
        fs::read_to_string(path).ok()?.trim().strip_prefix("0x")?,
        16,
    )
    .ok()
}

#[cfg(all(test, unix))]
mod sysfs_tests {
    use super::*;
    use std::path::PathBuf;

    struct Fixture {
        root: tempfile::TempDir,
    }

    impl Fixture {
        fn new() -> Self {
            let fixture = Self {
                root: tempfile::tempdir().unwrap(),
            };
            fixture.write("dev/kfd", "");
            fs::create_dir_all(fixture.path("sys/class/kfd/kfd/topology/nodes")).unwrap();
            fixture
        }

        fn path(&self, relative: &str) -> PathBuf {
            self.root.path().join(relative)
        }

        fn write(&self, relative: &str, contents: &str) {
            let path = self.path(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        }

        fn pci(&self, address: &str, vendor: &str, class: &str) {
            self.write(&format!("sys/bus/pci/devices/{address}/vendor"), vendor);
            self.write(&format!("sys/bus/pci/devices/{address}/class"), class);
        }

        fn node(&self, index: usize, address: &str, gpu_id: u32, name: &str, banks: &[(u32, u64)]) {
            let address: PciAddress = address.parse().unwrap();
            let root = format!("sys/class/kfd/kfd/topology/nodes/{index}");
            let location = (u16::from(address.bus) << 8)
                | (u16::from(address.device) << 3)
                | u16::from(address.function);
            self.write(&format!("{root}/gpu_id"), &gpu_id.to_string());
            self.write(&format!("{root}/name"), name);
            self.write(
                &format!("{root}/properties"),
                &format!(
                    "vendor_id 4098\nsimd_count 256\ndomain {}\nlocation_id {location}\nmem_banks_count {}\n",
                    address.domain,
                    banks.len()
                ),
            );
            for (bank, (heap_type, size)) in banks.iter().enumerate() {
                self.write(
                    &format!("{root}/mem_banks/{bank}/properties"),
                    &format!("heap_type {heap_type}\nsize_in_bytes {size}\n"),
                );
            }
        }

        fn detect(&self) -> Vec<Accelerator> {
            probe(&self.path("sys"), &self.path("dev/kfd"))
        }
    }

    #[test]
    fn admits_display_and_instinct_compute_devices_and_matches_each_node() {
        let fixture = Fixture::new();
        fixture.pci("0000:03:00.0", "0x1002\n", "0x030000\n");
        fixture.pci("0001:05:00.0", "0x1002\n", "0x120000\n");
        fixture.write(
            "sys/bus/pci/devices/0000:03:00.0/mem_info_vram_total",
            &(24_u64 << 30).to_string(),
        );
        fixture.write(
            "sys/bus/pci/devices/0001:05:00.0/mem_info_vram_total",
            &(192_u64 << 30).to_string(),
        );
        // KFD node order deliberately disagrees with physical PCI order.
        fixture.node(1, "0001:05:00.0", 22, "gfx942\n", &[(1, 192 << 30)]);
        fixture.node(
            2,
            "0000:03:00.0",
            11,
            "gfx1100\n",
            &[(0, 64 << 30), (1, 8 << 30), (2, 16 << 30), (4, 65536)],
        );
        let devices = fixture.detect();
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].device.to_string(), "rocm-pci:0000:03:00.0");
        assert_eq!(devices[0].name, "gfx1100");
        assert_eq!(devices[0].total_memory_bytes, Some(24 << 30));
        assert_eq!(devices[1].device.to_string(), "rocm-pci:0001:05:00.0");
        assert_eq!(devices[1].name, "gfx942");
        assert_eq!(devices[1].total_memory_bytes, Some(192 << 30));
        assert_eq!(
            serde_json::to_value(&devices[0]).unwrap(),
            serde_json::json!({
                "device": "rocm-pci:0000:03:00.0",
                "name": "gfx1100",
                "total_memory_bytes": 24_u64 << 30,
            })
        );
    }

    #[test]
    fn skips_cpu_nodes_unsupported_amd_devices_and_non_gpu_pci_classes() {
        let fixture = Fixture::new();
        fixture.pci("0000:01:00.0", "0x1002", "0x030000");
        fixture.pci("0000:02:00.0", "0x1002", "0x030000");
        fixture.node(1, "0000:02:00.0", 0, "cpu", &[(0, 64 << 30)]);
        fixture.pci("0000:03:00.0", "0x1002", "0x020000");
        fixture.node(2, "0000:03:00.0", 2, "network", &[]);
        fixture.pci("0000:04:00.0", "0x10de", "0x030000");
        fixture.node(3, "0000:04:00.0", 3, "other vendor", &[]);
        assert!(fixture.detect().is_empty());
    }

    #[test]
    fn missing_kfd_or_topology_reports_no_rocm_devices() {
        let fixture = Fixture::new();
        fixture.pci("0000:03:00.0", "0x1002", "0x120000");
        fixture.node(1, "0000:03:00.0", 1, "gfx942", &[]);
        fs::remove_file(fixture.path("dev/kfd")).unwrap();
        assert!(fixture.detect().is_empty());
        fixture.write("dev/kfd", "");
        fs::remove_dir_all(fixture.path("sys/class/kfd/kfd/topology")).unwrap();
        assert!(fixture.detect().is_empty());
    }

    #[test]
    fn missing_name_falls_back_and_partition_banks_do_not_establish_card_memory() {
        let fixture = Fixture::new();
        fixture.pci("0000:03:00.0", "0x1002", "0x030000");
        fixture.node(1, "0000:03:00.0", 1, "", &[(1, 8 << 30), (2, 16 << 30)]);
        let devices = fixture.detect();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "AMD GPU (0000:03:00.0)");
        assert_eq!(devices[0].total_memory_bytes, None);
    }

    #[test]
    fn malformed_or_unmatched_kfd_pci_properties_do_not_invent_devices() {
        let fixture = Fixture::new();
        fixture.pci("0000:03:00.0", "0x1002", "0x030000");
        fixture.node(1, "0000:04:00.0", 1, "unmatched", &[]);
        fixture.node(2, "0000:03:00.0", 2, "malformed", &[]);
        fixture.write(
            "sys/class/kfd/kfd/topology/nodes/2/properties",
            "vendor_id 4098\nsimd_count 256\ndomain 65536\nlocation_id 768\n",
        );
        assert!(fixture.detect().is_empty());
    }

    #[test]
    fn unreadable_drm_mapping_does_not_attribute_a_partition_to_another_function() {
        let fixture = Fixture::new();
        fixture.pci("0000:03:00.1", "0x1002", "0x120000");
        fixture.node(1, "0000:03:00.1", 1, "partition", &[]);
        let path = "sys/class/kfd/kfd/topology/nodes/1/properties";
        let mut properties = fs::read_to_string(fixture.path(path)).unwrap();
        properties.push_str("drm_render_minor 128\n");
        fixture.write(path, &properties);
        assert!(fixture.detect().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn partitions_resolve_via_drm_to_one_physical_device_and_whole_card_memory() {
        let fixture = Fixture::new();
        fixture.pci("0000:03:00.0", "0x1002", "0x120000");
        fixture.node(1, "0000:03:00.0", 1, "gfx942", &[(1, 96 << 30)]);
        fixture.node(2, "0000:03:00.1", 2, "gfx942", &[(1, 96 << 30)]);
        for (node, minor) in [(1, 128), (2, 129)] {
            let properties = format!("sys/class/kfd/kfd/topology/nodes/{node}/properties");
            let mut value = fs::read_to_string(fixture.path(&properties)).unwrap();
            value.push_str(&format!("drm_render_minor {minor}\n"));
            fixture.write(&properties, &value);
            let render = fixture.path(&format!("sys/class/drm/renderD{minor}"));
            fs::create_dir_all(&render).unwrap();
            std::os::unix::fs::symlink(
                fixture.path("sys/bus/pci/devices/0000:03:00.0"),
                render.join("device"),
            )
            .unwrap();
        }
        let devices = fixture.detect();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].device.to_string(), "rocm-pci:0000:03:00.0");
        assert_eq!(devices[0].total_memory_bytes, None);
        fixture.write(
            "sys/bus/pci/devices/0000:03:00.0/mem_info_vram_total",
            &(192_u64 << 30).to_string(),
        );
        assert_eq!(fixture.detect()[0].total_memory_bytes, Some(192 << 30));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_address_validates_shape_ranges_and_canonicalizes_case() {
        assert_eq!(
            "00AF:AB:1F.7".parse::<PciAddress>().unwrap().to_string(),
            "00af:ab:1f.7"
        );
        for invalid in [
            "",
            "03:00.0",
            "0000:03:20.0",
            "0000:03:00.8",
            "0000:03:00.g",
            "+000:03:00.0",
            "0000/03:00.0",
            "é000:03:0.0",
        ] {
            assert!(invalid.parse::<PciAddress>().is_err(), "{invalid}");
        }
    }

    #[test]
    fn accelerator_json_uses_a_physical_pci_identity() {
        let accelerator = crate::Accelerator {
            device: crate::DeviceType::Rocm {
                pci_address: "0001:05:00.0".parse().unwrap(),
            },
            name: "gfx942".to_owned(),
            total_memory_bytes: Some(192 << 30),
        };
        assert_eq!(
            serde_json::to_value(accelerator).unwrap(),
            serde_json::json!({
                "device": "rocm-pci:0001:05:00.0",
                "name": "gfx942",
                "total_memory_bytes": 192_u64 << 30,
            }),
        );
    }
}
