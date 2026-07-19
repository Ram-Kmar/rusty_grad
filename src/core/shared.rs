#[cfg(feature = "single_thread")]
pub type Shared<T> = std::rc::Rc<T>;

#[cfg(feature = "single_thread")]
pub fn new_shared<T>(data: T) -> Shared<T> {
    std::rc::Rc::new(data)
}

#[cfg(not(feature = "single_thread"))]
pub type Shared<T> = std::sync::Arc<T>;

#[cfg(not(feature = "single_thread"))]
pub fn new_shared<T>(data: T) -> Shared<T> {
    std::sync::Arc::new(data)
}

