from tei_client import HttpClient,GrpcClient
import time


HTTP_URLS = ['http://localhost:8080', 'http://localhost:8082', 'http://localhost:8086']
GRPC_URLS = ['localhost:8081', 'localhost:8083', 'localhost:8085']

if __name__ == '__main__':
	while len(HTTP_URLS) > 0:
		for url in HTTP_URLS:
			client = HttpClient(url)
			if client.health():
				print('http', url, 'is healthy')
				HTTP_URLS.remove(url)
		time.sleep(2)

	while len(GRPC_URLS) > 0:
		for url in GRPC_URLS:
			client = GrpcClient(url)
			if client.health():
				print('grpc', url, 'is healthy')
				GRPC_URLS.remove(url)
		time.sleep(2)
	
	print('all services are healthy')
	
