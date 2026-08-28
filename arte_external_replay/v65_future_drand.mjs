import fetchPolyfill from 'isomorphic-fetch'
import AbortControllerPolyfill from 'abort-controller'
import { fetchBeacon, HttpChainClient, HttpCachingChain } from 'drand-client'

globalThis.fetch = fetchPolyfill
globalThis.AbortController = AbortControllerPolyfill
const chainHash='8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce'
const publicKey='868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801af31'
const options={disableBeaconVerification:false,noCache:true,chainVerificationParams:{chainHash,publicKey}}
const client=new HttpChainClient(new HttpCachingChain('https://api.drand.sh',options),options)
const floor=await fetchBeacon(client)
let beacon=floor
for(let i=0;i<25 && beacon.round<=floor.round;i++){
  await new Promise(r=>setTimeout(r,3000))
  beacon=await fetchBeacon(client)
}
if(beacon.round<=floor.round) throw new Error(`NO_FUTURE_ROUND floor=${floor.round}`)
console.log(JSON.stringify({network:'drand-default',chain_hash:chainHash,public_key:publicKey,floor_round:floor.round,round:beacon.round,randomness:beacon.randomness,signature:beacon.signature,previous_signature:beacon.previous_signature,cryptographic_verification_enabled:true,target_entropy_strictly_future:true}))
