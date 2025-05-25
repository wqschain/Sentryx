import subprocess
import sys
import pkg_resources
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_dependencies():
    """Verify that all required packages are installed with correct versions"""
    required_packages = {
        'transformers': '4.30.0',
        'torch': '2.0.0',
        'numpy': '1.24.0',
        'tqdm': '4.65.0',
        'scikit-learn': '1.0.2',
        'datasets': '2.12.0'
    }
    
    missing_packages = []
    outdated_packages = []
    
    for package, min_version in required_packages.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            logger.info(f"✓ {package} version {installed_version} is installed")
            if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                outdated_packages.append((package, installed_version, min_version))
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning("Missing packages:")
        for package in missing_packages:
            logger.warning(f"✗ {package}")
    
    if outdated_packages:
        logger.warning("\nOutdated packages:")
        for package, installed, required in outdated_packages:
            logger.warning(f"✗ {package} version {installed} is older than required {required}")
    
    return len(missing_packages) == 0 and len(outdated_packages) == 0

def install_dependencies():
    """Install or upgrade required packages"""
    logger.info("Installing/upgrading required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "../../requirements.txt"])
        logger.info("✓ Successfully installed all dependencies")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install dependencies: {e}")
        return False

if __name__ == "__main__":
    logger.info("Verifying Python environment...")
    logger.info(f"Python version: {sys.version}")
    
    if not verify_dependencies():
        logger.warning("\nAttempting to install missing/outdated packages...")
        if install_dependencies():
            logger.info("\nVerifying after installation...")
            verify_dependencies()
    else:
        logger.info("\nAll dependencies are correctly installed! 🎉") 