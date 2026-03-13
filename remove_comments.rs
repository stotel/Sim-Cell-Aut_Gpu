use regex::Regex;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

fn main() {
    let re = Regex::new(r"
    
    let root = std::env::current_dir().expect("Failed to get current directory");
    
    for entry in WalkDir::new(&root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
        .filter(|e| !e.path().to_string_lossy().contains("target"))
    {
        let path = entry.path();
        
        match fs::read_to_string(&path) {
            Ok(content) => {
                let modified = re.replace_all(&content, "").to_string();
                
                if content != modified {
                    match fs::write(&path, &modified) {
                        Ok(_) => println!("Processed: {}", path.display()),
                        Err(e) => eprintln!("Error writing {}: {}", path.display(), e),
                    }
                }
            }
            Err(e) => eprintln!("Error reading {}: {}", path.display(), e),
        }
    }
    
    println!("Done!");
}
