import { writable } from 'svelte/store';

// Будем хранить текущее значение placeholder
export const placeholder = writable('');
