import fetchPolyfill from 'isomorphic-fetch'
import AbortControllerPolyfill from 'abort-controller'
import { fetchBeacon, HttpChainClient, HttpCachingChain } from 'drand-client'

globalThis.fetch = fetchPolyfill
globalThis.AbortController = AbortControllerPolyfill

const chainHash = '8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce'
const publicKey = '868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801af31'
const options = {
  disableBeaconVerification: false,
  noCache: true,
  chainVerificationParams: { chainHash, publicKey }
}
const chain = new HttpCachingChain('https://api.drand.sh', options)
const client = new HttpChainClient(chain, options)
const first = await fetchBeacon(client)
console.log(JSON.stringify({
  network: 'drand-default',
  chain_hash: chainHash,
  public_key: publicKey,
  round: first.round,
  randomness: first.randomness,
  signature: first.signature,
  previous_signature: first.previous_signature,
  cryptographic_verification_enabled: true
}))
