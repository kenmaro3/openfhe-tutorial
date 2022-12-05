#define PROFILE

#include "openfhe.h"

#include "stdio.h"

using namespace lbcrypto;
using namespace std;

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
        x.push_back(i);
    }
    size_t encodedLength = x.size();

    Plaintext ptxt = cryptoContext->MakeCKKSPackedPlaintext(x, 1, 0);

    ptxt->SetLength(encodedLength);
    std::cout << "Input: " << ptxt << std::endl;

    // Encryption
    Ciphertext<DCRTPoly> c = cryptoContext->Encrypt(keyPair.publicKey, ptxt);

    // Evaluation
    Ciphertext<DCRTPoly> c_add = cryptoContext->EvalAdd(c, c);
    Ciphertext<DCRTPoly> c_mul = cryptoContext->EvalMultAndRelinearize(c, c);

    // Bootstrapping
    auto pbs_c     = cryptoContext->EvalBootstrap(c);
    auto pbs_c_add = cryptoContext->EvalBootstrap(c_add);
    auto pbs_c_mul = cryptoContext->EvalBootstrap(c_mul);

    std::cout << "level remaind pbs_c: " << depth - pbs_c->GetLevel() << std::endl << std::endl;
    std::cout << "level remaind pbs_c_add: " << depth - pbs_c_add->GetLevel() << std::endl << std::endl;
    std::cout << "level remaind pbs_c_mul: " << depth - pbs_c_mul->GetLevel() << std::endl << std::endl;

    // Decryption
    Plaintext res_pbs_c;
    cryptoContext->Decrypt(keyPair.secretKey, pbs_c, &res_pbs_c);
    res_pbs_c->SetLength(encodedLength);
    std::cout << "res_pbs_c\n\t" << res_pbs_c << std::endl;

    Plaintext res_pbs_c_add;
    cryptoContext->Decrypt(keyPair.secretKey, pbs_c_add, &res_pbs_c_add);
    res_pbs_c_add->SetLength(encodedLength);
    std::cout << "res_pbs_c_add\n\t" << res_pbs_c_add << std::endl;

    Plaintext res_pbs_c_mul;
    cryptoContext->Decrypt(keyPair.secretKey, pbs_c_mul, &res_pbs_c_mul);
    res_pbs_c_mul->SetLength(encodedLength);
    std::cout << "res_pbs_c_mul\n\t" << res_pbs_c_mul << std::endl;
}
