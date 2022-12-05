#define PROFILE

#include "openfhe.h"

#include "stdio.h"

using namespace lbcrypto;
using namespace std;

using namespace std::chrono;
inline double get_time_msec(void) {
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()) / 1000000;
}

void SimpleBootstrapExample();

double relu(double x) {
    if (x >= 0) {
        return x;
    }
    else {
        return 0;
    }
}

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

    usint ringDim = cryptoContext->GetRingDimension();
    // This is the maximum number of slots that can be used for full packing.
    usint numSlots = ringDim / 2;
    std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl << std::endl;

    cryptoContext->EvalBootstrapSetup(levelBudget);

    auto keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

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
    Ciphertext<DCRTPoly> c = cryptoContext->Encrypt(keyPair.publicKey, ptxt);

    // relu
    double start = get_time_msec();
    auto c_relu  = cryptoContext->EvalChebyshevFunction(relu, c, -10, 10, 50);
    double end   = get_time_msec();

    // Bootstrapping
    auto pbs_c = cryptoContext->EvalBootstrap(c);

    auto pbs_c_relu = cryptoContext->EvalBootstrap(c_relu);

    std::cout << "level remaind pbs_c: " << depth - pbs_c->GetLevel() << std::endl << std::endl;
    std::cout << "level remaind pbs_c_relu: " << depth - pbs_c_relu->GetLevel() << std::endl << std::endl;

    // Decryption
    Plaintext res_pbs_c;
    cryptoContext->Decrypt(keyPair.secretKey, pbs_c, &res_pbs_c);
    res_pbs_c->SetLength(encodedLength);
    std::cout << "res_pbs_c\n\t" << res_pbs_c << std::endl;

    Plaintext res_pbs_c_relu;
    cryptoContext->Decrypt(keyPair.secretKey, pbs_c_relu, &res_pbs_c_relu);
    res_pbs_c_relu->SetLength(encodedLength);
    std::cout << "res_pbs_c_relu\n\t" << res_pbs_c_relu << std::endl;

    printf("approx_time: %f\n", end - start);
}
