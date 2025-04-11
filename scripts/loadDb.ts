//comments added for my understanding no need to follow them

import { DataAPIClient, Db } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "langchain/document_loaders/web/puppeteer";
import { OpenAI } from "openai";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import "dotenv/config";
import puppeteer from "puppeteer";

type SimilarityMetric = "cosine" | "dot_product" | "euclidean"; //used to determine how similar 2 vectors are (used for finding relative chinks)

//dot product faster than cosine but requires normalized vectors
//euclidean is the most accurate but slowest no need for normalization - find out how close 2 vectors are - small euclidean vectors means they are close in the vector space

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  OPEN_AI_KEY,
} = process.env;

const openai = new OpenAI({ apiKey: OPEN_AI_KEY });

//all the links we wanna scrape from

const f1Data = ["https://en.wikipedia.org/wiki/Formula_One"];

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_ENDPOINT, { namespace: ASTRA_DB_NAMESPACE });

//splitting into chunks to conserve tokens by only using the relative chunks by using the recursive character chunk text splitter

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512, //number of characters in each shunk
  chunkOverlap: 100, //number of characters to overlap between chunks
});

//create the collection here instead of in the astra datastacks database

const createCollection = async (
  similarityMetric: SimilarityMetric = "dot_product"
) => {
  const res = await db.createCollection(ASTRA_DB_COLLECTION, {
    vector: {
      dimension: 1536, //dimention size needs to match the model here(openai)
      metric: similarityMetric, //cosine, dot_product, euclidean
    },
  });
  console.log(res);
};

//allows us to get all the rlelevant data from the page and convert it into a format that can be used by the vector database

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const loadSampleData = async () => {
  const collection = await db.collection(ASTRA_DB_COLLECTION);
  for (const url of f1Data) {
    try {
      const content = await scrapePage(url);
      const chunks = await splitter.splitText(content);
      for (const chunk of chunks) {
        await delay(50000); // Add a 2-second delay between requests
        try {
          const embedding = await openai.embeddings.create({
            model: "text-embedding-3-small",
            input: chunk,
            encoding_format: "float",
          });
          const vector = embedding.data[0].embedding;
          const res = await collection.insertOne({
            $vector: vector,
            text: chunk,
          });
          console.log(res);
        } catch (err) {
          console.error("Error creating embedding:", err.message);
        }
      }
    } catch (err) {
      console.error("Error processing URL:", url, err.message);
    }
  }
};

//scrape the page using puppeteer and return the content of the page as a string

const scrapePage = async (url: string) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: true,
    },
    gotoOptions: {
      waitUntil: "domcontentloaded",
    },
    evaluate: async (page) => {
      //wait for the page to load and return the content of the page as a string
      const result = await page.evaluate(() => document.body.innerText); //wait for the page to load and return the content of the page as a string
      const browserInstance = await puppeteer.launch({ headless: true }); //launch the browser instance
      await browserInstance.close(); //close the browser instance
      return result;
    },
  });
  return (await loader.scrape())?.replace(/<[^>]*>?/gm, ""); //remove all html tags from the content
};

createCollection().then(() => {
  loadSampleData();
});
