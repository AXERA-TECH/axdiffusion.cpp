// HTTP protocol adapters. Each adapter only converts between its wire format and axd::Request / axd::Result;
// all generation goes through the JobQueue.
#pragma once

#include <string>
#include <vector>

#include "httplib.h"
#include "job_queue.hpp"
#include "json.hpp"

namespace axd {

struct ServerContext {
    ModelPlugin &model;
    JobQueue &queue;
    std::string out_dir;     // images returned by URL
    std::string model_name;  // id reported to clients
};

void register_openai(httplib::Server &srv, ServerContext &ctx);  // /v1/models, /v1/images/*, /v1/loras, /files/*
void register_a1111(httplib::Server &srv, ServerContext &ctx);   // /sdapi/v1/*
void register_jobs(httplib::Server &srv, ServerContext &ctx);    // /v1/jobs (async, progress via SSE)
void register_web(httplib::Server &srv, ServerContext &ctx);     // / and /health

// ---- shared helpers (api_util.cpp)
std::string base64_encode(const std::string &in);
std::string base64_decode(const std::string &in);  // accepts data: URIs
void json_error(httplib::Response &res, int status, const std::string &msg);
uint64_t resolve_seed(int64_t seed);                // -1 -> random
void parse_size(const std::string &s, int &w, int &h);  // "1024x1024"
std::vector<LoraSpec> loras_from_json(const nlohmann::json &j);
std::string strip_lora_tags(const std::string &prompt, std::vector<LoraSpec> &loras);  // A1111 <lora:name:w>
// Runs a request through the queue and waits; throws on failure / cancellation.
Result run_sync(ServerContext &ctx, Request req);
nlohmann::json result_info(const Request &req, const Result &r);
std::string png_of(const Image &img);
std::string save_png(ServerContext &ctx, const std::string &png);  // returns file name under out_dir

}  // namespace axd
