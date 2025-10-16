 
#[link(name = "vector", kind = "static")]
extern "C" {
    pub fn add_vectors_wrapper(a: *const i32, b: *const i32, c: *mut i32, n: i32);
}
