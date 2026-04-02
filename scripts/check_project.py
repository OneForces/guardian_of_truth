from guardian_truth.config import ensure_project_dirs, PROJECT_ROOT

def main():
    ensure_project_dirs()
    print('OK', PROJECT_ROOT)

if __name__ == '__main__':
    main()
