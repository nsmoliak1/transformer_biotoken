import os

def print_py_files_info(project_root):
    #print(f"Scanning project root: {project_root}")  # Debug print
    # Walk through the project directory
    for dirpath, dirnames, filenames in os.walk(project_root):
        #print(f"Currently scanning directory: {dirpath}")
        # Exclude certain directories if necessary
        dirnames[:] = [d for d in dirnames if d not in ('venv', '__pycache__', '.git')]

        for filename in filenames:
            #print(f"Found file: {filename}")
            if filename.endswith('.py'):
                #print(f"Processing Python file: {filename}")
                # Get the full path and relative path
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, project_root)

                # Read the first 12 lines of the file
                try:
                    with open(full_path, 'r', encoding='utf-8') as file:
                        first_12_lines = []
                        for _ in range(12):
                            line = file.readline()
                            if not line:
                                break
                            first_12_lines.append(line.rstrip('\n'))
                except Exception as e:
                    print(f"Error reading file {relative_path}: {e}")
                    continue  # Skip to the next file

                # Print the information
                print(f"\nFile Title: {filename}")
                print(f"Location: {relative_path}")
                print("First 12 lines:")
                for line in first_12_lines:
                    print(line)
                print("\n" + "-" * 50 + "\n")
            else:
                print(f"Skipping non-Python file: {filename}")

if __name__ == '__main__':
    # Replace 'path_to_BioTokenNLS' with the actual path to your BioTokenNLS directory
    project_root = r'C:\Users\2860909S\BioTokens\BioTokenNLS'
    print(f"Project root is set to: {project_root}")
    print_py_files_info(project_root)
