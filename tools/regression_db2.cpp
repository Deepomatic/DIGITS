// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>

#include <sstream>
#include <string>
#include <iterator>
#include <iomanip>


#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "opencv2/opencv.hpp"
#include <sys/stat.h>

#include <memory>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
  //::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << argc;
  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/regression_db");
    return 1;
  }
  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::string buffer;

  std::string db_name(argv[3]);

  bool flag = true;
  std::vector<std::pair<int, std::vector<std::string>>> lines;
  std::string type;
  while (getline(infile, buffer)) {
      std::istringstream iss(buffer);
      std::string tmp;
      std::vector<std::string> tokens;
      int i = -1;
      int idx;
      while (getline(iss, tmp, ' ' )) {
          if (++i == 0) {
            type = tmp;
          }
          else if (i == 1) {
            idx = std::atof(tmp.c_str());
          }
          else {
            tokens.push_back(tmp);
          }
      }
      lines.emplace_back(std::pair<int, std::vector<std::string>>(idx, tokens));
    }
 //    std::string tmp;
 //    std::vector<float> line = {};
 //    if (flag) {
 //      LOG(INFO) << buffer;
 //      while (getline(iss, tmp, ' ' )) {
 //        int val = std::atoi(tmp.c_str());
 //        if (val > 0) {
 //          labels.push_back(val);
 //        }
 //      }
 //      flag = false;
 //    }
 //    else {
 //      getline(iss, tmp, ' ');
 //      std::pair<std::string, std::vector<float>> new_input = std::make_pair(tmp, line);
 //      while (getline(iss, tmp, ' ' )) {
 //        try {
 //          new_input.second.emplace_back(std::stold(tmp));
 //        }
 //        catch (const std::invalid_argument &e) {
 //           LOG(WARNING) << "error on line[" << buffer << "]-->" <<  tmp << "<--";
 //        }
 //      }
 //      lines.emplace_back(new_input);
 //    }
 //  }
 // //  [0.3948019742965698, 0.3316831588745117, 0.7660890817642212, 0.5581682920455933]
 // //[0.3308550185873606 0.33663366336633666 0.701363073110285 0.5631188118811881]
 //
 //  // if (FLAGS_shuffle) {
 //  //   // randomly shuffle data
 //  //   LOG(INFO) << "Shuffling data";
 //  //   shuffle(lines.begin(), lines.end());
 //  // }
 //  LOG(INFO) << "A total of " << lines.size() << " images.";
 //
 //  if (encode_type.size() && !encoded)
 //    LOG(INFO) << "encode_type specified, assuming encoded=true.";
 //
 //
  //Create new DB
  std::string root_output = argv[1] + std::string("/") + std::string(db_name);
  if(mkdir(root_output.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    LOG(WARNING) << "can't create:" << root_output;
  }
 //
  std::unique_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(root_output + "/data_lmdb", db::NEW);
  std::unique_ptr<db::Transaction> txn(db->NewTransaction());

  if (type == "data") { //ASSUMING IT'S ONLY IMAGES
    int resize_height = std::max<int>(0, FLAGS_resize_height);
    int resize_width = std::max<int>(0, FLAGS_resize_width);
    int count = 0;
    for (const auto &line : lines) {
      LOG(INFO) << line.first << " " << line.second[0];
      Datum datum;
      const int kMaxKeyLength = 256;
      char key_cstr[kMaxKeyLength];
      int data_size = 0;
      bool data_size_initialized = false;
      bool status;
      std::string enc = encode_type;
      if (encoded && !enc.size()) {
        // Guess the encoding type from the file name
        string fn = line.second[0];
        size_t p = fn.rfind('.');
        if ( p == fn.npos )
         LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
        enc = fn.substr(p);
        std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
      }
      status = ReadImageToDatum(line.second[0], 0, resize_height, resize_width, is_color, enc, &datum);
      // datum.set_channels(3);
      // datum.set_width(resize_width);
      // datum.set_height(resize_height);
      // datum.set_label(0);
      if (status == false) continue;
      if (check_size) {
        if (!data_size_initialized) {
         data_size = datum.channels() * datum.height() * datum.width();
         data_size_initialized = true;
        } else {
         const std::string& data = datum.data();
         CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
             << data.size();
        }
      }
      // sequential
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d", line.first);
      // Put in db
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(string(key_cstr, length), out);
      LOG(INFO) << "Processed " << count << "/" << lines.size();

      if (count++ % 1000 == 0) {
      txn->Commit();
      txn.reset(db->NewTransaction());
      }
    }
    txn->Commit();
    LOG(INFO) << "Total images added: " << count;
    LOG(INFO) << "--Done--";
  }
  else if (type == "float_data") {
    std::string out;
    int count = 0;
    for (const auto &line : lines) {
      Datum datum;
      datum.set_channels(line.second.size());
      datum.set_height(1);
      datum.set_width(1);
      datum.set_label(0);
      datum.set_encoded(false);
      int idx = line.first;
      const int kMaxKeyLength = 256;
      char key_cstr[kMaxKeyLength];
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d", idx);
      for (const string &val: line.second) {
        datum.add_float_data(std::stof(val));
      }
      out.clear();
      datum.SerializeToString(&out);
      txn->Put(std::string(key_cstr, length), out);
      if (count++ % 1000 == 0) {
        txn->Commit();
        txn.reset(db->NewTransaction());
      }
      LOG(INFO) << "Processed " << count << "/" << lines.size();
    }
    txn->Commit();
    LOG(INFO) << "Total images added: " << count;
    LOG(INFO) << "--Done--";
  }
  else {
    return -1;
  }
  return 0;
}
