#define PROFILE

#include "openfhe.h"

#include "stdio.h"
#include <random>
#include <vector>
#include <cassert>

using namespace lbcrypto;
using namespace std;

using namespace std::chrono;
inline double get_time_msec(void) {
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()) / 1000000;
}

void SimpleBootstrapExample();

int main(int argc, char* argv[]) {
    SimpleBootstrapExample();
}

void SimpleBootstrapExample() {
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

    uint32_t levelsUsedBeforeBootstrap = 12;
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
    cryptoContext->Enable(PRE);

    usint ringDim = cryptoContext->GetRingDimension();
    // This is the maximum number of slots that can be used for full packing.
    usint numSlots = ringDim / 2;
    std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl << std::endl;

    cryptoContext->EvalBootstrapSetup(levelBudget);

    auto keyPair1 = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair1.secretKey);
    cryptoContext->EvalBootstrapKeyGen(keyPair1.secretKey, numSlots);

    KeyPair<DCRTPoly> keyPair2 = cryptoContext->KeyGen();
    EvalKey<DCRTPoly> evalKey  = cryptoContext->ReKeyGen(keyPair1.secretKey, keyPair2.publicKey);

    // Making plaintext vector
    std::vector<double> x;
    for (int i = 0; i < 10; i++) {
        x.push_back(i - 5);
    }
    size_t encodedLength = x.size();

    Plaintext ptxt = cryptoContext->MakeCKKSPackedPlaintext(x, 1, 0);

    ptxt->SetLength(encodedLength);
    std::cout << "Input: " << ptxt << std::endl;

    // Encryption
    Ciphertext<DCRTPoly> c = cryptoContext->Encrypt(keyPair1.publicKey, ptxt);

    // some operation
    Ciphertext<DCRTPoly> c_mul = cryptoContext->EvalMultAndRelinearize(c, c);

    // bootstrap operation
    Ciphertext<DCRTPoly> c_bs = cryptoContext->EvalBootstrap(c);

    // ReEncryption
    Ciphertext<DCRTPoly> c_re     = cryptoContext->ReEncrypt(c, evalKey);
    Ciphertext<DCRTPoly> c_mul_re = cryptoContext->ReEncrypt(c_mul, evalKey);
    Ciphertext<DCRTPoly> c_bs_re = cryptoContext->ReEncrypt(c_bs, evalKey);

    double start = get_time_msec();
    Plaintext res_c;
    cryptoContext->Decrypt(keyPair2.secretKey, c_re, &res_c);
    res_c->SetLength(encodedLength);
    std::cout << "res_c\n\t" << res_c << std::endl;

    Plaintext res_c_mul;
    cryptoContext->Decrypt(keyPair2.secretKey, c_mul_re, &res_c_mul);
    res_c_mul->SetLength(encodedLength);
    std::cout << "res_c_mul\n\t" << res_c_mul << std::endl;

    Plaintext res_c_bs;
    cryptoContext->Decrypt(keyPair2.secretKey, c_bs_re, &res_c_bs);
    res_c_bs->SetLength(encodedLength);
    std::cout << "res_c_bs\n\t" << res_c_bs << std::endl;

    double end = get_time_msec();

    printf("total_time: %f\n", end - start);
}
