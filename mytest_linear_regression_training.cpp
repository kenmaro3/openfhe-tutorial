#define PROFILE

#include "openfhe.h"

#include "stdio.h"
#include <random>
#include <vector>
#include <cassert>

using namespace lbcrypto;
using namespace std;

constexpr int MIN = -1;
constexpr int MAX = 1;

using namespace std::chrono;
inline double get_time_msec(void) {
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()) / 1000000;
}

std::random_device rd;
std::default_random_engine eng(rd());
std::uniform_real_distribution<double> distr(MIN, MAX);

struct Struct_key_and_context {
    KeyPair<DCRTPoly> keys;
    CryptoContext<DCRTPoly> context;
};

struct Struct_VVVD_VVD {
    vector<vector<vector<double>>> xs;
    vector<vector<double>> ys;
};

struct Struct_data_for_linear_regression {
    vector<vector<double>> xs;
    vector<double> ys;
    vector<double> w;
    double b;
};

struct Struct_plain_w_b {
    vector<double> w;
    double b;
};

struct Struct_update_dw_db {
    Ciphertext<DCRTPoly> dw;
    Ciphertext<DCRTPoly> db;
};

struct Struct_enc_w_b {
    Ciphertext<DCRTPoly> w;
    Ciphertext<DCRTPoly> b;
    Ciphertext<DCRTPoly> loss;
};

Struct_plain_w_b initialize_w_b(int attrs) {
    vector<double> w(attrs);
    double b = distr(eng);

    for (int i = 0; i < attrs; i++) {
        w[i] = distr(eng);
    };

    Struct_plain_w_b res = {w, b};

    return res;
}

Struct_VVVD_VVD make_mini_batches(vector<vector<double>> x, vector<double> y, int batch_size) {
    std::random_device randomDevice;
    std::vector<uint32_t> randomSeedVector(10);

    std::generate(randomSeedVector.begin(), randomSeedVector.end(), std::ref(randomDevice));
    std::seed_seq randomSeed(randomSeedVector.begin(), randomSeedVector.end());
    std::mt19937 mt(randomSeed);
    int32_t* index;
    auto x_old = x;
    auto y_old = y;
    vector<vector<vector<double>>> x_batch;
    vector<vector<double>> y_batch;

    index = new int32_t[x.size()];
    for (int32_t i = 0; i < x.size(); i++) {
        index[i] = i;
    }
    std::shuffle(index, index + x.size(), mt);
    for (int32_t i = 0; i < x.size(); i++) {
        x[i] = x[index[i]];
        y[i] = y[index[i]];
    }
    for (int32_t i = 0; i < int32_t(x.size() / batch_size); i++) {
        x_batch.emplace_back();
        y_batch.emplace_back();
        for (int32_t j = 0; j < batch_size; j++) {
            x_batch[i].push_back(x[j + i * batch_size]);
            y_batch[i].push_back(y[j + i * batch_size]);
        }
    }

    Struct_VVVD_VVD result = {x_batch, y_batch};
    return result;
}

void print_vector(vector<double> xs) {
    for (int i = 0; i < xs.size(); i++) {
        printf("%f, ", xs[i]);
    }
    printf("\n");
}

void print_matrix(vector<vector<double>> x) {
    for (int i = 0; i < x.size(); i++) {
        print_vector(x[i]);
    }
    printf("\n");
}

void print_vector(vector<double> xs, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f, ", xs[i]);
    }
    printf("\n");
}

void print_matrix(vector<vector<double>> x, int size) {
    for (int i = 0; i < x.size(); i++) {
        print_vector(x[i], size);
    }
    printf("\n");
}

double get_random() {
    return distr(eng);
}

vector<double> get_random_vector(int size) {
    vector<double> res(size);
    for (int i = 0; i < size; i++) {
        res[i] = distr(eng);
    }
    return res;
}

vector<vector<double>> get_random_matrix(int attrs, int ds) {
    vector<vector<double>> res;
    for (int i = 0; i < ds; i++) {
        res.push_back(get_random_vector(attrs));
    }
    return res;
}

vector<double> calculate_y(vector<vector<double>> xs, vector<double> w, double b) {
    vector<double> res;
    for (int i = 0; i < xs.size(); i++) {
        double tmp = 0;
        for (int j = 0; j < xs[i].size(); j++) {
            tmp += xs[i][j] * w[j];
        }
        tmp += b;
        res.push_back(tmp);
    }
    return res;
}

vector<double> flatten_xs(vector<vector<double>> xs) {
    vector<double> res(xs.size() * xs[0].size());
    for (int i = 0; i < xs.size(); i++) {
        for (int j = 0; j < xs[i].size(); j++) {
            res[i * xs[i].size() + j] = xs[i][j];
        }
    }
    return res;
}

vector<double> pp_y(vector<double> y, int attrs, int bs) {
    assert(y.size() == bs);
    vector<double> res(attrs * bs);
    for (int i = 0; i < bs; i++) {
        res[i * attrs] = y[i];
    }
    return res;
}

vector<double> pp_w(vector<double> w, int bs) {
    vector<vector<double>> tmp;
    for (int i = 0; i < bs; i++) {
        tmp.emplace_back(w);
    }
    return flatten_xs(tmp);
}

vector<double> pp_b(double b, int attrs, int bs) {
    vector<double> res(attrs * bs);
    for (int i = 0; i < bs; i++) {
        res[i * attrs] = b;
    }
    return res;
}

Ciphertext<DCRTPoly> rotate_and_copy_right(Ciphertext<DCRTPoly>& x, CryptoContext<DCRTPoly>& cc, int attrs) {
    auto res = x;
    for (int i = 1; i < attrs; i++) {
        res = cc->EvalAdd(res, cc->EvalRotate(x, (-1) * i));
    }
    return res;
}

Ciphertext<DCRTPoly> rotate_and_copy_left(Ciphertext<DCRTPoly>& x, CryptoContext<DCRTPoly>& cc, int attrs) {
    auto tmp = x;
    auto res = x;
    for (int i = 1; i < attrs; i++) {
        cc->EvalRotate(tmp, (-1) * i);
        res = cc->EvalAdd(res, tmp);
    }
    return res;
}

Ciphertext<DCRTPoly> rotate_and_copy_skip(Ciphertext<DCRTPoly>& x, CryptoContext<DCRTPoly>& cc, int bs, int attrs) {
    // copy bs times, by skipping attrs
    auto tmp = x;
    auto res = x;
    for (int i = 1; i < bs; i++) {
        res = cc->EvalAdd(res, cc->EvalRotate(x, (-1) * i * attrs));
    }
    return res;
}

Struct_enc_w_b pp_enc_w_b(Struct_enc_w_b& w_b, CryptoContext<DCRTPoly>& cc, int attrs, int bs) {
    // w = (w0, w1, ..., wattrs-1, 0, 0, ..., 0)
    // b = (b, 0, 0, ..., 0)
    auto ppd_enc_w = rotate_and_copy_skip(w_b.w, cc, bs, attrs);
    auto ppd_enc_b = rotate_and_copy_skip(w_b.b, cc, bs, attrs);

    Struct_enc_w_b ppd_enc_w_b = {ppd_enc_w, ppd_enc_b};

    return ppd_enc_w_b;
}

Ciphertext<DCRTPoly> encrypt(vector<double>& x, Struct_key_and_context& kc) {
    Plaintext ptxt         = kc.context->MakeCKKSPackedPlaintext(x, 1, 0);
    Ciphertext<DCRTPoly> c = kc.context->Encrypt(kc.keys.publicKey, ptxt);
    return c;
}

vector<double> decrypt(Ciphertext<DCRTPoly>& c, Struct_key_and_context& kc, int size) {
    // Decryption
    Plaintext res_c;
    kc.context->Decrypt(kc.keys.secretKey, c, &res_c);
    res_c->SetLength(size);
    vector<double> packed_value = res_c.get()->GetRealPackedValue();
    //std::cout << "res_c\n\t" << res_c << std::endl;
    return packed_value;
}

Struct_enc_w_b apply_bootstrap_w_b(Struct_enc_w_b& w_b, CryptoContext<DCRTPoly>& cc) {
    auto bsd_enc_w     = cc->EvalBootstrap(w_b.w);
    auto bsd_enc_b     = cc->EvalBootstrap(w_b.b);
    Struct_enc_w_b res = {bsd_enc_w, bsd_enc_b};

    return res;
}

Struct_enc_w_b fake_bootstrap_w_b(Struct_enc_w_b& w_b, Struct_key_and_context& kc, int attrs, int bs) {
    vector<double> dec_w = decrypt(w_b.w, kc, attrs * bs);
    vector<double> dec_b = decrypt(w_b.b, kc, attrs * bs);

    auto enc_w         = encrypt(dec_w, kc);
    auto enc_b         = encrypt(dec_b, kc);
    Struct_enc_w_b res = {enc_w, enc_b};

    return res;
}

Ciphertext<DCRTPoly> rotate_and_sum(Ciphertext<DCRTPoly>& x, int& attrs, CryptoContext<DCRTPoly>& cc) {
    auto tmp = x;
    for (int i = 1; i < attrs; i++) {
        tmp = cc->EvalAdd(tmp, cc->EvalRotate(x, i));
    }

    return tmp;
}

Ciphertext<DCRTPoly> rotate_and_sum_skip(Ciphertext<DCRTPoly>& x, int& sum_num, int skip_num,
                                         CryptoContext<DCRTPoly>& cc) {
    auto tmp = x;
    for (int i = 1; i < sum_num; i++) {
        tmp = cc->EvalAdd(tmp, cc->EvalRotate(x, skip_num * i));
    }

    return tmp;
}

vector<double> get_unit1(int attrs, int bs) {
    vector<double> res(attrs * bs);
    for (int i = 0; i < bs; i++) {
        res[attrs * i] = 1.0;
    }
    return res;
}

vector<double> get_unit2(int attrs) {
    vector<double> res(attrs);
    for (int i = 0; i < attrs; i++) {
        res[i] = 1.0;
    }
    return res;
}

vector<double> get_unit3() {
    vector<double> res;
    res.emplace_back(1.0);
    return res;
}

Ciphertext<DCRTPoly> calc_loss(Ciphertext<DCRTPoly>& y_hat, Ciphertext<DCRTPoly>& y, Struct_key_and_context& kc,
                               int attrs, int bs) {
    auto tmp1 = kc.context->EvalSub(y_hat, y);
    auto tmp2 = kc.context->EvalMultAndRelinearize(tmp1, tmp1);
    auto tmp3 = rotate_and_sum_skip(tmp2, bs, attrs, kc.context);
    return tmp3;
}

Struct_update_dw_db backward(Ciphertext<DCRTPoly>& y_hat, Ciphertext<DCRTPoly>& y, Ciphertext<DCRTPoly>& x,
                             Struct_key_and_context& kc, int attrs, int bs, double lr) {
    // y_hat = (y_hat0, 0, 0, .. , y_hat1, 0, 0, ...)

    auto tmp1 = kc.context->EvalSub(y_hat, y);

    // cout << "debug tmp1" << endl;
    // auto dec_tmp1 = decrypt(tmp1, kc, attrs * bs);
    // print_vector(dec_tmp1, attrs * bs);

    // dL/dw
    auto tmp2 = rotate_and_copy_right(tmp1, kc.context, attrs);
    // cout << "debug tmp2" << endl;
    // auto dec_tmp2 = decrypt(tmp2, kc, attrs * bs);
    // print_vector(dec_tmp2, attrs * bs);

    auto tmp3 = kc.context->EvalMultAndRelinearize(tmp2, x);
    // cout << "debug tmp3" << endl;
    // auto dec_tmp3 = decrypt(tmp3, kc, attrs * bs);
    // print_vector(dec_tmp3, attrs * bs);

    auto tmp4 = rotate_and_sum_skip(tmp3, bs, attrs, kc.context);
    // cout << "debug tmp4" << endl;
    // auto dec_tmp4 = decrypt(tmp4, kc, attrs * bs);
    // print_vector(dec_tmp4, attrs * bs);

    auto tmp5 = kc.context->EvalMult(tmp4, 1.0 / double(bs));
    // cout << "debug tmp5" << endl;
    // auto dec_tmp5 = decrypt(tmp5, kc, attrs * bs);
    // print_vector(dec_tmp5, attrs * bs);

    auto tmp6 = kc.context->EvalMult(tmp5, lr);
    // cout << "debug tmp6" << endl;
    // auto dec_tmp6 = decrypt(tmp6, kc, attrs * bs);
    // print_vector(dec_tmp6, attrs * bs);

    vector<double> unit2 = get_unit2(attrs);
    auto enc_unit2       = encrypt(unit2, kc);

    auto tmp7 = kc.context->EvalMultAndRelinearize(tmp6, enc_unit2);
    // cout << "debug tmp7" << endl;
    // auto dec_tmp7 = decrypt(tmp7, kc, attrs * bs);
    // print_vector(dec_tmp7, attrs * bs);
    // tmp7 = (dw0, dw1, ..., dw_attrs-1)

    // dL/db
    auto tmp_b_1         = rotate_and_sum_skip(tmp1, bs, attrs, kc.context);
    auto tmp_b_2         = kc.context->EvalMult(tmp_b_1, 1.0 / double(bs));
    auto tmp_b_3         = kc.context->EvalMult(tmp_b_2, lr);
    vector<double> unit3 = get_unit3();
    auto enc_unit3       = encrypt(unit3, kc);
    auto tmp_b_4         = kc.context->EvalMultAndRelinearize(tmp_b_3, enc_unit3);
    //auto tmp_b_5         = rotate_and_copy_skip(tmp_b_4, kc.context, bs, attrs);

    Struct_update_dw_db res = {tmp7, tmp_b_4};
    return res;
}

Ciphertext<DCRTPoly> forward(Ciphertext<DCRTPoly>& x, Struct_enc_w_b& w_b, Struct_key_and_context& kc, int attrs,
                             int bs) {
    auto tmp1 = kc.context->EvalMultAndRelinearize(x, w_b.w);
    auto tmp2 = rotate_and_sum(tmp1, attrs, kc.context);
    auto tmp3 = kc.context->EvalAdd(tmp2, w_b.b);

    vector<double> unit1 = get_unit1(attrs, bs);

    auto enc_unit1 = encrypt(unit1, kc);

    auto tmp4 = kc.context->EvalMultAndRelinearize(tmp3, enc_unit1);

    // (y_hat0, 0, 0, .., y_hat1, 0, 0, ...)

    return tmp4;
}

Struct_enc_w_b update(Struct_enc_w_b& w_b, Struct_update_dw_db& dw_db, Struct_key_and_context& kc, int attrs, int bs) {
    auto ppd_dw = rotate_and_copy_skip(dw_db.dw, kc.context, bs, attrs);
    auto ppd_db = rotate_and_copy_skip(dw_db.db, kc.context, bs, attrs);

    auto new_w = kc.context->EvalSub(w_b.w, ppd_dw);
    auto new_b = kc.context->EvalSub(w_b.b, ppd_db);

    Struct_enc_w_b res = {new_w, new_b};
    return res;
}

Struct_enc_w_b main_batch(Ciphertext<DCRTPoly>& x, Ciphertext<DCRTPoly>& y, Struct_enc_w_b& w_b,
                          Struct_key_and_context& kc, int attrs, int bs, double lr) {
    auto y_hat = forward(x, w_b, kc, attrs, bs);

    // cout << "debug yhat===========" << endl;
    // auto tmp_dec_yhat = decrypt(y_hat, kc, attrs * bs);
    // print_vector(tmp_dec_yhat, attrs * bs);

    auto loss = calc_loss(y_hat, y, kc, attrs, bs);
    // cout << "debug loss===========" << endl;
    // auto tmp_dec_loss = decrypt(loss, kc, attrs * bs);
    // print_vector(tmp_dec_loss, attrs * bs);

    Struct_update_dw_db dw_db = backward(y_hat, y, x, kc, attrs, bs, lr);
    // cout << "debug dw===========" << endl;
    // auto tmp_dec_dw = decrypt(dw_db.dw, kc, attrs * bs);
    // print_vector(tmp_dec_dw, attrs * bs);

    // cout << "debug db===========" << endl;
    // auto tmp_dec_db = decrypt(dw_db.db, kc, attrs * bs);
    // print_vector(tmp_dec_db, attrs * bs);

    Struct_enc_w_b new_w_b = update(w_b, dw_db, kc, attrs, bs);
    // cout << "debug new_w===========" << endl;
    // auto tmp_dec_new_w = decrypt(new_w_b.w, kc, attrs * bs);
    // print_vector(tmp_dec_new_w, attrs * bs);
    // cout << "debug new_b===========" << endl;
    // auto tmp_dec_new_b = decrypt(new_w_b.b, kc, attrs * bs);
    // print_vector(tmp_dec_new_b, attrs * bs);

    new_w_b.loss = loss;
    return new_w_b;
}

Struct_key_and_context initialize_context_and_keys(int attrs, int bs) {
    CCParams<CryptoContextCKKSRNS> parameters;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);

#if NATIVEINT == 128 && !defined(__EMSCRIPTEN__)
    ScalingTechnique rescaleTech = FIXEDAUTO;
    usint dcrtBits               = 78;
    usint firstMod               = 89;
#else
    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    usint dcrtBits               = 59;
    usint firstMod               = 60;
#endif

    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(rescaleTech);
    parameters.SetFirstModSize(firstMod);

    std::vector<uint32_t> levelBudget = {4, 4};
    uint32_t approxBootstrapDepth     = 8;

    uint32_t levelsUsedBeforeBootstrap = 8;
    usint depth =
        levelsUsedBeforeBootstrap + FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, levelBudget, secretKeyDist);
    parameters.SetMultiplicativeDepth(depth);

    printf("this is my depth %d\n", depth);

    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);
    cryptoContext->Enable(ADVANCEDSHE);
    cryptoContext->Enable(FHE);

    usint ringDim = cryptoContext->GetRingDimension();
    // This is the maximum number of slots that can be used for full packing.
    usint numSlots = ringDim / 2;
    std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl << std::endl;

    cryptoContext->EvalBootstrapSetup(levelBudget);

    auto keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    vector<int> tmp;
    for (int i = (-1) * attrs * bs; i < attrs * bs; i++) {
        tmp.emplace_back(i);
    }
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, tmp);

    Struct_key_and_context res = {keyPair, cryptoContext};
    return res;
}


Struct_data_for_linear_regression prepare_linear_regression(int attrs, int ds, int bs);

int main(int argc, char* argv[]) {

    int attrs = 10;
    int ds    = 100;
    int bs    = 10;
    double lr = 0.05;
    int epoch = 30;

    double start, stop;

    start                     = get_time_msec();
    Struct_key_and_context kc = initialize_context_and_keys(attrs, bs);
    stop                      = get_time_msec();
    printf("time_initialze_keys_and_context: %f\n", stop - start);

    Struct_data_for_linear_regression datas = prepare_linear_regression(attrs, ds, bs);

    Struct_plain_w_b initial_w_b = initialize_w_b(attrs);
    printf("initialized w_b: \n");
    print_vector(initial_w_b.w, attrs);
    printf("%f\n", initial_w_b.b);

    // initialize w_b and then preprocess then encrypt
    vector<double> ppd_w = pp_w(initial_w_b.w, bs);
    vector<double> ppd_b = pp_b(initial_w_b.b, attrs, bs);
    auto enc_w           = encrypt(ppd_w, kc);
    auto enc_b           = encrypt(ppd_b, kc);
    Struct_enc_w_b w_b   = {enc_w, enc_b};

    Ciphertext<DCRTPoly> loss;

    vector<double> loss_list;

    double whole_start, whole_stop;

    whole_start = get_time_msec();

    for (int k = 0; k < epoch; k++) {
        Struct_VVVD_VVD batched_data = make_mini_batches(datas.xs, datas.ys, bs);

        for (int i = 0; i < batched_data.xs.size(); i++) {
            printf("epoch: %d, batch %d/%ld\n", k, i, batched_data.xs.size());
            vector<double> batch_x     = flatten_xs(batched_data.xs[i]);
            vector<double> batch_y     = batched_data.ys[i];
            vector<double> ppd_batch_y = pp_y(batch_y, attrs, bs);

            auto enc_xs = encrypt(batch_x, kc);
            auto enc_ys = encrypt(ppd_batch_y, kc);
            //printf("enc done\n");

            start = get_time_msec();
            w_b   = main_batch(enc_xs, enc_ys, w_b, kc, attrs, bs, lr);
            stop  = get_time_msec();
            loss  = w_b.loss;
            //printf("time_main_batch: %f\n", stop - start);

            // cout << "debug dec_w" << endl;
            // cout << "debug dec_b" << endl;

            // auto tmp_dec_w = decrypt(w_b.w, kc, attrs * bs);
            // auto tmp_dec_b = decrypt(w_b.b, kc, attrs * bs);

            // print_vector(tmp_dec_w, bs * attrs);
            // print_vector(tmp_dec_b, bs * attrs);
            // cout << "===============\n" << endl;

            //printf("main_batch done\n");

            // w_b = pp_enc_w_b(w_b, kc.context, attrs, bs);
            // printf("pp_enc_w_b done\n");

            start = get_time_msec();
            w_b   = apply_bootstrap_w_b(w_b, kc.context);
            //w_b  = fake_bootstrap_w_b(w_b, kc, attrs, bs);
            stop = get_time_msec();
            //printf("bs_enc_w_b done\n");
            printf("time_bootstrap_w_b: %f\n", stop - start);

            //auto dec_w    = decrypt(w_b.w, kc, attrs * bs);
            //auto dec_b    = decrypt(w_b.b, kc, attrs * bs);
            auto dec_loss = decrypt(loss, kc, bs);

            //printf("\n\n=================================\n");
            //printf("dec_w: \n");
            //print_vector(dec_w);
            //printf("w: \n");
            //print_vector(datas.w);

            //printf("\ndec_b: \n");
            //print_vector(dec_b);
            //printf("b: \n");

            //printf("\n%f\n", datas.b);
            //printf("loss: %f\n", dec_loss[0]);

            loss_list.emplace_back(dec_loss[0]);
            print_vector(loss_list);
        }
    }

    printf("\nloss_list:\n");
    print_vector(loss_list);

    auto dec_w = decrypt(w_b.w, kc, attrs * bs);
    auto dec_b = decrypt(w_b.b, kc, attrs * bs);

    printf("\n\n=================================\n");
    printf("dec_w: \n");
    print_vector(dec_w);
    printf("w: \n");
    print_vector(datas.w);

    printf("\ndec_b: \n");
    print_vector(dec_b);
    printf("b: \n");

    whole_stop = get_time_msec();
    printf("time_whole: %f\n", whole_stop - whole_start);
}

Struct_data_for_linear_regression prepare_linear_regression(int attrs, int ds, int bs) {
    vector<double> w = get_random_vector(attrs);
    double b         = get_random();
    vector<vector<double>> xs = get_random_matrix(attrs, ds);
    vector<double> ys = calculate_y(xs, w, b);

    Struct_data_for_linear_regression res = {xs, ys, w, b};
    return res;
}
