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

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/regression_db");
    return 1;
  }
  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, std::vector<float> > > lines;
  std::vector<float> labels;
  std::string buffer;

  bool flag = true;
  while (getline(infile, buffer)) {
    std::istringstream iss(buffer);
    std::string tmp;
    std::vector<float> line = {};
    if (flag) {
      LOG(INFO) << buffer;
      while (getline(iss, tmp, ' ' )) {
        int val = std::atoi(tmp.c_str());
        if (val > 0) {
          labels.push_back(val);
        }
      }
      flag = false;
    }
    else {
      getline(iss, tmp, ' ');
      std::pair<std::string, std::vector<float>> new_input = std::make_pair(tmp, line);
      while (getline(iss, tmp, ' ' )) {
        try {
          new_input.second.emplace_back(std::stold(tmp));
        }
        catch (const std::invalid_argument &e) {
           LOG(WARNING) << "error on line[" << buffer << "]-->" <<  tmp << "<--";
        }
      }
      lines.emplace_back(new_input);
    }
  }
 //  [0.3948019742965698, 0.3316831588745117, 0.7660890817642212, 0.5581682920455933]
 //[0.3308550185873606 0.33663366336633666 0.701363073110285 0.5631188118811881]

  // if (FLAGS_shuffle) {
  //   // randomly shuffle data
  //   LOG(INFO) << "Shuffling data";
  //   shuffle(lines.begin(), lines.end());
  // }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  //Create new DB
  std::string root_output = argv[1] + std::string("/") + std::string(argv[3]);
  if(mkdir(root_output.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    LOG(WARNING) << "can't create:" << root_output;
  }

  std::unique_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(root_output + "/data_lmdb", db::NEW);
  std::unique_ptr<db::Transaction> txn(db->NewTransaction());

  std::map<int, std::pair<std::unique_ptr<db::DB>, std::unique_ptr<db::Transaction>>> labels_db;
  for (int i = 0;i < labels.size(); i++) {
    std::unique_ptr<db::DB> tmp(db::GetDB(FLAGS_backend));
    tmp->Open(root_output + "/labels_" + std::to_string(i) + "_lmdb" , db::NEW);
    std::unique_ptr<db::Transaction>  tmp2(tmp->NewTransaction());
    labels_db[i] = std::make_pair<decltype(tmp), decltype(tmp2)>(std::move(tmp), std::move(tmp2));
  }

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;
  int nbr_images = lines.size();

  for (int line_id = 0; line_id < nbr_images; ++line_id) {
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadImageToDatum(lines[line_id].first,
        0, resize_height, resize_width, is_color,
        enc, &datum);
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
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d", line_id);
    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (count % 1000 == 0) {
      txn->Commit();
      txn.reset(db->NewTransaction());
    }

    int idx = 0;
    int label_count = 0;
    int max_size = labels[idx];

    Datum _datum;
    _datum.set_channels(max_size);
    _datum.set_height(1);
    _datum.set_width(1);
    _datum.set_label(0);
    _datum.set_encoded(false);
    for (const float &val : lines[line_id].second) {
      if (label_count++ < max_size) {
        _datum.add_float_data(val);
      }
      else {
        _datum.SerializeToString(&out);
        labels_db[idx].second->Put(std::string(key_cstr, length), out);
        //reset
        out.clear();
        idx += 1;
        label_count = 1;
        max_size = labels[idx];
        _datum.Clear();
        _datum.set_channels(max_size);
        _datum.set_height(1);
        _datum.set_width(1);
        _datum.set_label(0);
        _datum.set_encoded(false);
        _datum.add_float_data(val);
      }
    }
    for (float val : _datum.float_data()) {
      LOG(INFO) << val;
    }
    LOG(FATAL) << key_cstr;
    _datum.SerializeToString(&out);
    labels_db[idx].second->Put(std::string(key_cstr, length), out);
    //out.clear();
    //_datum.SerializeToString(&out);
    //txn_label->Put(std::string(key_cstr, length), out);

    if (count % 1000 == 0) {
      for (auto &p : labels_db) {
        p.second.second->Commit();
        p.second.second.reset(p.second.first->NewTransaction());
      }
      //txn_label->Commit();
      //txn_label.reset(db_label->NewTransaction());
    }
    count++;

    LOG(INFO) << "Processed " << line_id << "/" << nbr_images;
  }
  // write the last batch
  if (count % 1000 != 0) {
    for (auto &p : labels_db) {
      p.second.second->Commit();
    }
    txn->Commit();
  }
  LOG(INFO) << "Total images added: " << count;
  LOG(INFO) << "--Done--";
  return 0;
}
