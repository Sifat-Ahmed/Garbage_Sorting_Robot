import socket


try:

    HOST = '192.168.1.3'   # The server's hostname or IP address
    PORT = 3002  # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b"Hello, world")
        data = s.recv(1024)


except KeyboardInterrupt:
    server.close()