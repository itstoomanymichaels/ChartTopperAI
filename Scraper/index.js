const fs = require('fs');
const fastcsv = require('fast-csv');
const SpotifyWebApi = require('spotify-web-api-node');

//Scraper Backend Vars
const uniqueTracksSet = new Set();
const configOptions = './config.json';
const filePath = './songs.csv';
let lastRecordedId = 0;
const spotifyApi = new SpotifyWebApi({
  clientId: configOptions.clientIDd,
  clientSecret: configOptions.sercret
});

//CSV Constants
const headers = [
  "id", "track_name", "album_name", "artist", "genre", "popularity",
  "duration_ms", "acousticness", "explicit", "danceability", "energy",
  "key", "loudness", "mode", "speechiness", "instrumentalness",
  "valence", "tempo", "time_signature",
];

//Rate Limiting Constants
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
let genres = null;
const BATCH = 50;
const WRITE_DELAY = 200;
let songBuffer = [];

const authenticate = async () => {
  try {
    const auth = await spotifyApi.clientCredentialsGrant();
    spotifyApi.setAccessToken(auth.body['access_token']);
    console.log(`Succesfully Logged Into Spotify`);
  } catch (err) {
    console.error(`Login Error: ${err.message}`);
  }
}

const bulkWrite = async () => {
  if (songBuffer.length === 0) return;
  const writeHeaders = !fs.existsSync(filePath) || fs.statSync(filePath).size === 0;
  
  const formattedSongs = songBuffer.map(song => {
    return headers.reduce((acc, header) => {
      acc[header] = song[header] || '';
      return acc;
    }, {});
  });

  // Use fast-csv to write the data
  const ws = fs.createWriteStream(filePath, { flags: 'a' });
  await new Promise((resolve, reject) => {
    fastcsv
      .write(formattedSongs, { headers: writeHeaders, includeEndRowDelimiter: true })
      .on('finish', async () => {
        songBuffer = [];
        // Wait until thread catches up
        await saveConfig();
        resolve(); 
      })
      .on('error', (err) => {
        console.error('Error writing to CSV:', err);
        reject(err); 
      })
      .pipe(ws);
  });
  //Sleep to prevent 30 second rate limit
  await sleep(WRITE_DELAY);
}
//Write to the config file to save line number
const saveConfig = async () => {
  const configJSON = JSON.parse(await fs.promises.readFile(configOptions, 'utf-8'));
  configJSON.currId = lastRecordedId;
  await fs.promises.writeFile(configOptions, JSON.stringify(configJSON, null, 2));
}

//Returns 50 songs in a playlist
const randomSong = async () => {
  try {
    const { id: playlistId, genre } = await getRandomPlaylist();
    const data = await spotifyApi.getPlaylistTracks(playlistId);
    const validTracks = data.body.items.map(item => item.track).filter(track => track !== null);
    return { tracks: validTracks, trackIds: validTracks.map(track => track.id), genre: genre };
  } catch (err) {
    console.error('Error fetching tracks:', err);
  }
}

const getRandomPlaylist = async () => {
  try {
    let failure = true;
    let playlists = null;
    let data = null;
    let randomGenre = null;

    while (failure) {
      randomGenre = genres[Math.floor(Math.random() * genres.length)];
      data = await spotifyApi.searchPlaylists(`genre:${randomGenre}`, { limit: 50 });
      //If genre default doesn't exist, go for a random user playlist with that name
      if (data.body.playlists.items.length === 0) {
        data = await spotifyApi.searchPlaylists(`playlist:${randomGenre}`, { limit: 50 });
      }
      if (data.body.playlists.items.length > 0) {
        failure = false;
      }
    }

    playlists = data.body.playlists.items;
    const randomPlaylist = playlists[Math.floor(Math.random() * playlists.length)];
    return { id: randomPlaylist.id, genre: randomGenre };
  } catch (err) {
    console.error('Error fetching random playlist:', err);
  }
}

const requestWithRateLimit = async (promiseFn) => {
  const maxRetries = 5;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await promiseFn();
    } catch (err) {
      if (err.statusCode === 429) {
        console.log(err.headers);
        const retryAfter = parseInt(err.headers['retry-after']) || Math.pow(4, attempt) * 10000; // Exponential backoff
        console.warn(`Rate limit exceeded. Retrying after ${retryAfter} ms`);
        await sleep(retryAfter);
      } else {
        throw err;
      }
    }
  }
  throw new Error('Max retries reached for Spotify API request.');
}

async function getTrackAnalytics(trackId) {
  return requestWithRateLimit(async () => {
    const [audioFeatures, audioAnalysis] = await Promise.all([
      spotifyApi.getAudioFeaturesForTracks(trackId),
    ]);
    return audioFeatures.body;
  });
}

/*
  Set Getter and Setters.
  Used to manage the CSV in memory
*/
async function loadUniqueTracks(filePath) {
  return new Promise((resolve, reject) => {
    fs.createReadStream(filePath)
      .pipe(fastcsv.parse({ headers: true }))
      .on('data', (row) => {
        // Use a unique key format: `${track_name}-${artist}`
        uniqueTracksSet.add(`${row.track_name}-${row.artist}`);
      })
      .on('end', () => {
        console.log(`Loaded ${uniqueTracksSet.size} unique tracks from CSV.`);
        resolve();
      })
      .on('error', (err) => 
        reject(err));
  });
}

function isTrackUnique(trackName, artistName) {
  const key = `${trackName}-${artistName}`;
  return !uniqueTracksSet.has(key); 
}

function addTrackToSet(trackName, artistName) {
  const key = `${trackName}-${artistName}`;
  uniqueTracksSet.add(key); 
}

// Main function to orchestrate the fetching and writing process
async function main() {
  await authenticate();
  const configJSON = JSON.parse(await fs.promises.readFile(configOptions, 'utf-8'));
  genres = configJSON.items;
  lastRecordedId = configJSON.currId;
  await loadUniqueTracks(filePath);

  let tries = 0;
  while (tries < 100000) {
    try {
      const { tracks, trackIds, genre } = await randomSong();
      const analytics = await getTrackAnalytics(trackIds);
      let currTrackNum = 0;
      // Keep the below the same, as it is the CSV format
      for (const randomTrack of tracks) {
        let trackAnalytics = analytics.audio_features[currTrackNum++];
        const filteredSongData = {
          id: ++lastRecordedId,
          track_name: randomTrack.name,
          album_name: randomTrack.album.name,
          artist: randomTrack.artists[0].name,
          genre: genre,
          popularity: randomTrack.popularity,
          duration_ms: randomTrack.duration_ms,
          acousticness: trackAnalytics.acousticness || 0,
          explicit: randomTrack.explicit,
          danceability: trackAnalytics.danceability,
          energy: trackAnalytics.energy,
          key: trackAnalytics.key,
          loudness: trackAnalytics.loudness,
          mode: trackAnalytics.mode,
          speechiness: trackAnalytics.speechiness,
          instrumentalness: trackAnalytics.instrumentalness,
          valence: trackAnalytics.valence,
          tempo: trackAnalytics.tempo,
          time_signature: trackAnalytics.time_signature
        };

        if (filteredSongData.track_name != null && isTrackUnique(randomTrack.name, randomTrack.artists[0].name)) {
          addTrackToSet(randomTrack.name, randomTrack.artists[0].name);
          songBuffer.push(filteredSongData);

          // Write to CSV in batches
          if (songBuffer.length >= BATCH) {
            await bulkWrite();
          }
        }
      }
    } catch (err) {
      console.warn(`An error occurred: ${err.message}`);
      await saveConfig();
    }
    tries++;
  }

  // Write any remaining songs in the buffer after the loop
  await bulkWrite();
}


main().catch(console.error);