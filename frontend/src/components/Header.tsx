import type { ReactNode } from 'react';

interface HeaderProps {
  title: string;
  description?: string;
  leftContent?: ReactNode;
  rightContent?: ReactNode;
  bottomContent?: ReactNode;
}

export const Header = ({
  title,
  description,
  leftContent,
  rightContent,
  bottomContent,
}: HeaderProps) => (
  <div className="bg-gray-800/40 backdrop-blur-lg rounded-xl p-6 mb-6 border border-cyan-500/30 shadow-lg shadow-cyan-500/10">
    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <h1 className="text-xl md:text-2xl font-bold text-white mb-2">
          {title}
        </h1>
        {description && (
          <p className="text-sm md:text-base text-cyan-300">{description}</p>
        )}
        {leftContent}
      </div>

      {rightContent && (
        <div className="text-xs md:text-base flex flex-col sm:flex-row gap-3">
          {rightContent}
        </div>
      )}
    </div>

    {bottomContent && (
      <div className="text-xs md:text-base mt-4">{bottomContent}</div>
    )}
  </div>
);
