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

const minRound = process.argv[2] ? Number(process.argv[2]) : null
let first = await fetchBeacon(client)
let beacon = first

const floor = minRound !== null ? minRound : first.round
for (let i = 0; i < 20 && beacon.round <= floor; i++) {
  await new Promise(resolve => setTimeout(resolve, 3000))
  beacon = await fetchBeacon(client)
}
if (beacon.round <= floor) {
  throw new Error(`NO_NEW_DRAND_ROUND: floor=${floor}, got=${beacon.round}`)
}
console.log(JSON.stringify({
  network: 'drand-default',
  chain_hash: chainHash,
  public_key: publicKey,
  round: beacon.round,
  randomness: beacon.randomness,
  signature: beacon.signature,
  previous_signature: beacon.previous_signature,
  cryptographic_verification_enabled: true,
  round_strictly_after_floor: true,
  floor_round: floor
}))
