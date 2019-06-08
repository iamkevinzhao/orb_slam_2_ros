#ifndef TCP_CLIENT_H
#define TCP_CLIENT_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netdb.h>
#include <vector>
#include <thread>
#include <mutex>

//using namespace std;

class TCPClient
{
  private:
    void send_thread();
    int sock;
    std::string address;
    int port;
    struct sockaddr_in server;
    std::thread thread_;
    bool exit_ = false;

    std::mutex buf_mutex_;
    std::string buf_;
  public:
    TCPClient();
    ~TCPClient();
    bool setup(std::string address, int port);
    bool Send(const std::string& data);
    bool block_send(std::string data);
    std::string receive(int size = 4096);
    std::string read();
    void exit();
};

#endif
