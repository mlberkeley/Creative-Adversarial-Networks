import boto3
import os

# s3 = boto3.resource('s3')
# s3.meta.client.upload_file('tmp/', BUCKET_NAME, 'tmp/')


def bucket_exists(bucket):
  s3 = boto3.resource('s3')
  return s3.Bucket(bucket) in s3.buckets.all()
def upload_path(local_directory, bucket, destination):
	client = boto3.client('s3')
	# enumerate local files recursively
	for root, dirs, files in os.walk(local_directory):

		for filename in files:

			# construct the full local path
			local_path = os.path.join(root, filename)

			# construct the full Dropbox path
			relative_path = os.path.relpath(local_path, local_directory)
			s3_path = os.path.join(destination, relative_path)

			# relative_path = os.path.relpath(os.path.join(root, filename))

			print('Searching "%s" in "%s"' % (s3_path, bucket))
			try:
				client.head_object(Bucket=bucket, Key=s3_path)
				# print("Path found on S3! Skipping %s..." % s3_path)
			except:
				print("Uploading %s..." % s3_path)
				client.upload_file(local_path, bucket, s3_path)
