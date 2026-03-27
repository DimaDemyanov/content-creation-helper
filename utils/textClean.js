const EMOJI_REGEX = /[\u{1F300}-\u{1FFFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu;
const HASHTAG_REGEX = /#\w+/g;
const URL_REGEX = /https?:\/\/\S+/g;
const SPECIAL_REGEX = /[^\p{L}\p{N}\s]/gu;
const MULTI_SPACE_REGEX = /\s+/g;

export function cleanText(text) {
  if (!text) return '';

  return text
    .replace(EMOJI_REGEX, ' ')
    .replace(HASHTAG_REGEX, ' ')
    .replace(URL_REGEX, ' ')
    .replace(SPECIAL_REGEX, ' ')
    .replace(MULTI_SPACE_REGEX, ' ')
    .trim();
}
