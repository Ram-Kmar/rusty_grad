use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind{
    #[default]
    Cpu,
    Metal,
}

impl BackendKind{
    pub fn as_str(self) -> &'static str{
        match self{
            BackendKind::Cpu => "cpu",
            BackendKind::Metal => "Metal",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BackendKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(BackendKind::Cpu),
            "metal" => Ok(BackendKind::Metal),
            other => Err(format!("unsupported backend '{other}'")),
        }
    }
}

