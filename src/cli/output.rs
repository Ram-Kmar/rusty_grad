pub fn print_section(title: &str, body: impl AsRef<str>, end_marker: &str){
    println!("{title}");
    println!("{}", body.as_ref());
    println!("{end_marker}");
}