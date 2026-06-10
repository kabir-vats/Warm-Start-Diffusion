"""Slightly adapted from Zihao Zhou"""

import fnmatch
import glob
import os
import time

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError

from src.utilities.utils import get_logger


log = get_logger(__name__)

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
if not S3_ENDPOINT_URL and not S3_BUCKET_NAME:
    S3_ENDPOINT_URL = os.environ["S3_ENDPOINT_URL"] = "https://rosedata.ucsd.edu"
    S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"] = "salva-data-and-results"
checks = ["S3_ENDPOINT_URL", "S3_BUCKET_NAME"]
for check in checks:
    if not os.getenv(check):
        raise EnvironmentError(f"Please set the {check} environment variable.")

# Export S3 credentials from ~/.config/s3
credentials_maybe_dir = os.path.expanduser("~/.config/s3")
if os.path.exists(credentials_maybe_dir):
    if os.environ.get("AWS_ACCESS_KEY_ID") is None:
        os.environ["AWS_ACCESS_KEY_ID"] = open(f"{credentials_maybe_dir}/access_key_id").read().strip()
    if os.environ.get("AWS_SECRET_ACCESS_KEY") is None:
        os.environ["AWS_SECRET_ACCESS_KEY"] = open(f"{credentials_maybe_dir}/secret_access_key").read().strip()
# Check if credentials are provided
if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
    config = Config(retries={"max_attempts": 5, "mode": "adaptive"}, max_pool_connections=50)
    # Credentials are provided, use them to create the client
    s3_client = boto3.client("s3", endpoint_url=S3_ENDPOINT_URL, config=config)
else:
    # Credentials are not provided, use anonymous access
    log.info("Using anonymous access to S3. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for authenticated access.")
    s3_client = boto3.client("s3", endpoint_url=S3_ENDPOINT_URL, config=Config(signature_version=UNSIGNED))


def get_local_files(s3_path, local_path):
    """
    Recursively get local files that match the s3_path pattern in the local_path directory.
    """
    wildcard_index = s3_path.find("*")
    if wildcard_index == -1:
        prefix = os.path.dirname(s3_path)
        pattern = os.path.basename(s3_path)
    else:
        prefix = s3_path[:wildcard_index]
        if "/" in prefix:
            prefix = os.path.dirname(prefix)
            pattern = s3_path[len(prefix) + 1 :]
        else:
            prefix = "."
            pattern = s3_path

    prefix = os.path.normpath(os.path.join(local_path, prefix))
    local_files = glob.glob(prefix + "/**", recursive=True)

    filtered_local_files = []
    for file in local_files:
        if os.path.isdir(file):
            continue
        file = os.path.normpath(file)
        if pattern:
            if fnmatch.fnmatch(os.path.relpath(file, prefix), pattern):
                file = os.path.normpath(file)
                filtered_local_files.append(file)
        else:
            filtered_local_files.append(file)
    return filtered_local_files


def get_s3_objects(s3_path):
    """
    Recursively get all objects in S3 bucket that match the s3_path pattern.
    """
    wildcard_index = s3_path.find("*")
    if wildcard_index == -1:
        prefix = s3_path
        pattern = ""
    else:
        prefix = s3_path[:wildcard_index]
        if prefix.endswith("/"):
            prefix = prefix[:-1]
            pattern = s3_path[len(prefix) + 1 :]
        else:
            pattern = s3_path[len(prefix) :]

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix)

    filtered_s3_objects = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if pattern:  # only apply fnmatch if there's a pattern to match
                if fnmatch.fnmatch(key[len(prefix) :], pattern):
                    filtered_s3_objects.append(key)
            else:
                filtered_s3_objects.append(key)
    return filtered_s3_objects


def download_s3_objects(s3_objects, local_path="./"):
    """
    Download specified S3 objects to the local file system.
    """
    for s3_key in s3_objects:
        # Construct the full local filepath
        local_file_path = os.path.join(local_path, s3_key)

        # Create directory if it doesn't exist
        local_file_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_file_dir):
            os.makedirs(local_file_dir)

        download_s3_object(s3_key, local_file_path)


def exists_s3_object(s3_file_path):
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_file_path)
        return True
    except ClientError:
        return False


def download_s3_object(s3_file_path, local_file_path: str, throw_error: bool = True):
    # Download the file from S3
    try:
        if os.path.exists(local_file_path):
            log.info(f"File {local_file_path} already exists")
            return
        # Make sure the directory exists if it has a directory structure
        if os.path.dirname(local_file_path) != "":
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the file from S3
        s3_client.download_file(S3_BUCKET_NAME, s3_file_path, local_file_path)

        # Verify file was downloaded successfully
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Failed to download file to {local_file_path}")

        log.info(f"Downloaded {s3_file_path} to {local_file_path}")
    except ClientError as e:
        log.warning(f"Failed to download {s3_file_path}: {e}")
        # List all files in the directory
        try:
            s3_objects = get_s3_objects(os.path.dirname(s3_file_path))
        except ClientError as e2:
            s3_objects = "<Failed to list files in directory>"
            log.warning(f"Failed to list files in directory: {e2}")
        if throw_error:
            raise ValueError(f"File {s3_file_path} not found in S3. Files in directory: {s3_objects}") from e
        else:
            log.info(f"Files in directory: {s3_objects}")
            return s3_objects


def download_s3_path(s3_path, local_path="./"):
    """
    Download all files in the S3 path to the local file system.
    """
    s3_objects = get_s3_objects(s3_path)
    download_s3_objects(s3_objects, local_path)


def list_s3_objects(s3_path):
    """
    List all directories / files in S3 bucket under the given path.
    """
    prefix = s3_path.lstrip("/")

    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix, Delimiter="/")
    objects = []
    directories = []

    for page in page_iterator:
        directories.extend(page.get("CommonPrefixes", []))
        objects.extend(page.get("Contents", []))

    for d in directories:
        log.info(f"Directory: {d['Prefix']}")

    for obj in objects:
        log.info(f"File: {obj['Key']}")


def delete_local_files(local_files):
    """
    Delete local files.
    """
    for file in local_files:
        os.remove(file)
        log.info(f"Deleted {file}")
        ## Delete empty folders if any
        folder = os.path.dirname(file)
        if not os.listdir(folder):
            os.rmdir(folder)
            log.info(f"Deleted {folder}")


def print_folders(files):
    """
    Print folders of files, assuming the files are sorted by folder.
    """
    last_folder = None
    for file in files:
        folder = "/".join(file.split("/")[:-1]) + "/"
        if folder != last_folder:
            log.info(folder)
            last_folder = folder


def remove_s3_objects(objects_to_delete):
    """
    Remove objects in S3.
    """
    objects_to_delete = [{"Key": obj} for obj in objects_to_delete]
    if objects_to_delete:
        s3_client.delete_objects(Bucket=S3_BUCKET_NAME, Delete={"Objects": objects_to_delete})
        for obj in objects_to_delete:
            log.info(f"Removed {obj['Key']} from S3")


def remove_s3_path(s3_path):
    """
    Remove all files in the S3 path.
    """
    s3_objects = get_s3_objects(s3_path)
    remove_s3_objects(s3_objects)


def upload_s3_object(local_file_path, s3_file_path, force_upload: bool = True, retry=3, **kwargs):
    """
    Upload a single local file to S3.
    Args:
        local_file_path: The path to the local file.
        s3_file_path: The path to the S3 file. If it ends with a "/", the local file will be uploaded with the same name to that directory.
        force_upload: Whether to upload the file even if it already exists in S3.
        retry: The number of times to retry the upload.
    """
    assert os.path.isfile(local_file_path), f"{local_file_path} is not a file"
    if s3_file_path.endswith("/"):
        s3_file_path += os.path.basename(local_file_path)
    else:
        # Check that both path's have same file extension
        local_file_ext = os.path.splitext(local_file_path)[1]
        s3_file_ext = os.path.splitext(s3_file_path)[1]
        assert (
            local_file_ext == s3_file_ext
        ), f"File extensions do not match: {local_file_ext} != {s3_file_ext}. If you intended s3_filepath to be a directory, append a '/' to the end of it."

    if not force_upload and exists_s3_object(s3_file_path):
        log.info(f"Skipping {local_file_path} as it already exists in S3")
        return False
    for i in range(retry):
        try:
            s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_file_path, **kwargs)
            log.info(f"Uploaded {local_file_path} to {s3_file_path}")
            return True
        except Exception as e:
            log.warning(
                f"Failed to upload {local_file_path} with S3_BUCKET_NAME={S3_BUCKET_NAME} and s3_file_path={s3_file_path}: {e}"
            )
            # sleep for 10 seconds before retrying
            time.sleep(5)
            if i == retry - 1:
                raise e
    return False


def upload_s3_objects(local_files, local_path="./", s3_path=""):
    """
    Upload local files to S3.
    """
    for local_file in local_files:
        if os.path.isfile(local_file):
            s3_key = os.path.relpath(local_file, os.path.dirname(local_path))
            s3_key = os.path.normpath(s3_key)
            if s3_path:
                s3_key = os.path.join(s3_path, s3_key)
            try:
                s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                log.info(f"File {s3_key} already exists in S3")
            except ClientError:
                s3_client.upload_file(local_file, S3_BUCKET_NAME, s3_key)
                log.info(f"Uploaded {local_file} to {s3_key}")
        else:
            log.warning(f"Skipping {local_file} as it is not a file.")


def upload_s3_path(s3_path, local_path="./"):
    """
    Upload all files in the local path to the S3 path.
    """
    local_files = get_local_files(s3_path, local_path)
    upload_s3_objects(local_files, local_path)


def interactive_list_and_action(s3_path, local_path):
    """
    List local and S3 files/folders, then ask the user whether to
    - delete local files
    - upload local files
    - remove s3 files
    - download s3 files
    """
    if s3_path.endswith("/"):
        filetype = "folders"
        s3_path += "*"
    else:
        filetype = "files"

    log.info(f"Local {filetype} matching pattern:")
    local_files = get_local_files(s3_path, local_path)

    if filetype == "folders":
        print_folders(local_files)
    else:
        for file in local_files:
            log.info(file)

    log.info(f"\nS3 {filetype} matching pattern:")
    s3_objects = get_s3_objects(s3_path)

    if filetype == "folders":
        print_folders(s3_objects)
    else:
        for file in s3_objects:
            log.info(file)

    action = input("\nChoose an action [delete (local), remove (S3), download, upload, exit]: ").strip().lower()
    if action == "delete":
        delete_local_files(local_files)
    elif action == "upload":
        upload_s3_objects(local_files, local_path)
    elif action == "remove":
        remove_s3_objects(s3_objects)
    elif action == "download":
        download_s3_objects(s3_objects, local_path)
    elif action == "exit":
        pass
    else:
        log.info("Invalid action")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S3 utils for managing files and folders.")
    parser.add_argument("--find", help="Find S3 files", action="store_true")
    parser.add_argument("--list", help="List S3 files", action="store_true")
    parser.add_argument("--download", help="Download S3 files", action="store_true")
    parser.add_argument("--upload", help="Upload local files to S3", action="store_true")
    parser.add_argument("--remove", help="Remove S3 files", action="store_true")
    parser.add_argument("--delete", help="Delete local files", action="store_true")
    parser.add_argument("--interactive", help="Interactive mode", action="store_true")
    parser.add_argument("path", help="The S3 or local path pattern", type=str)

    args = parser.parse_args()

    s3_path = args.path
    local_path = "./"

    if args.find:
        file_type = "folders" if s3_path.endswith("/") else "files"
        s3_objects = get_s3_objects(s3_path + "**" if file_type == "folders" else s3_path)
        if file_type == "folders":
            print_folders(s3_objects)
        else:
            for obj in s3_objects:
                log.info(obj)
    elif args.list:
        list_s3_objects(s3_path)
    elif args.download:
        download_s3_path(s3_path, local_path)
    elif args.upload:
        upload_s3_path(s3_path, local_path)
    elif args.remove:
        remove_s3_path(s3_path)
    elif args.delete:
        local_files = get_local_files(s3_path, local_path)
        delete_local_files(local_files)
    elif args.interactive:
        interactive_list_and_action(s3_path, local_path)
    else:
        parser.print_help()
